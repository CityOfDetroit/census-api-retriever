import json
import os
import numpy as np
import pandas as pd
import requests
from functools import reduce
from typing import Any

def make_census_field_name(table_name: str, field_number: int, field_type='E') -> str:
    """
    Turns input parameters into census api style field names

    Args:
        table_name (str): Census Table Name. Example: "B01001".
        field_name (int): Field/Column Number in the census table. Will be z-filled to 3 digits.
        field_type (str, optional): E for Estimate; M for Margin of Error. Defaults to 'E'.

    Raises:
        ValueError: If "field_type" does not equal "E" or "M"

    Returns:
        str: A census formated field name
    """
    if field_type.upper() in ['E','M']:
        field_name = table_name + '_' + f'{field_number}'.zfill(3) + field_type.upper()
        return field_name
    else:
        raise ValueError('Parameter "field_type" must equal "E" or "M"')

def fill_param(url: str, url_params: dict) -> str:
    """
    Fills out URL parameters for a templated url string like "for=place:*&in=state:{state}"
    (here {state} is templated).

    Args:
        url (str): URL string
        url_params (dict): parameter keyss and values

    Returns:
        str: URL with completed parameters ("...={state}" -> "...=26")
    """
    if url_params:
        for param, value in url_params.items():
            url = url.replace("{"+param+"}", str(value))
    return url 

def generate_geoids(df: pd.DataFrame, geography: str) -> pd.Series:
    """
    Generate geoid column when given a geography type.

    Args:
        df (pd.DataFrame): Dataframe to use to compute geoid
        geography (str): Census geography type such as "tract" or "place"

    Raises:
        ValueError: If geography parameter is not valid
    
    Returns:
        pd.Series: Series with computed geoid
    """
    if geography == 'tract':
        return df.state.astype(str)+df.county.astype(str)+df.tract.astype(str)
    elif geography == 'place':
        return df.state.astype(str)+df.place.astype(str)
    elif geography == 'bg': # TODO: Make available
        return df.state.astype(str)+df.county.astype(str)+df.tract.astype(str)+df['block group'].astype(str)
    elif geography == 'zip':
        return df['zip code tabulation area'].astype(str)
    else:
        raise ValueError('Geography parameter is invalid.')


class CensusAPIRetriever:
    def __init__(self, request_config_path: str, geography_config_path='configs/setup/geographies.json', field_output_types = 'EM', calc_columns_only = False):
        """
        Class to get census data using the Census Data Config setup.

        Args:
            request_config_path (str): Path to request config json
            geography_config_path (str, optional): Path to geography config json. Defaults to 'configs/setup/geographies.json'.
            field_output_types (str, optional): field_output_types options: e|m|both; Defaults to 'both'.
            calc_columns_only (bool, optional): if true will keep only custom calc columns from the config
        """
        self.request_config_path = request_config_path
        self.geography_config_path = geography_config_path
        self.field_output_types = field_output_types
        self.calc_columns_only = calc_columns_only
        self.geography_index = 0

    # Configs

    def _initialize_requests(self):
        """
        Opens request config json
        """
        with open(self.request_config_path) as file:
            self.request_config = json.load(file)
    
    def _initialize_geographies(self):
        """
        Opens geography config json
        """
        with open(self.geography_config_path) as file:
            self.geography_config = json.load(file)

    def _initialize_configs(self):
        """
        Initializes configuration files
        """
        self._initialize_requests()
        self._initialize_geographies()

    # Utils
    
    def _get_request_fields(self) -> list[dict[str, Any]]:
        """
        Gets field list from request config
        Returns:
            list[dict]: List of fields in request containing dicts with src_id (int), table (str), and fields (list[int])
        """
        return self.request_config["fields"]
    
    def _geographic_level_is_custom(self) -> bool:
        """
        Determines if a geographic level is custom by seeing if there is a key called 
        "level" in the geographic config dict for it.

        Returns:
            bool: If geographic level is custom (uses apportionment of a census geography)
        """
        return bool(self.geography_config[self.request_config["geography"][self.geography_index]["level"]].get("level"))
    
    def _process_multiple_geographies(self) -> bool:
        return bool(self.request_config.get("geography")[self.geography_index].get('geog_type_field'))
    
    def _get_geographic_level(self) -> dict:
        """
        Gets the geographic level to retreive from the api, either at face value, or if custom, 
        for further approtionment into a custom geography. For example, a custom geography of 
        neighborhoods (nh) will request tracts to aggregate into neighborhoods later. 
        
        If the requested geography is custom, "level" and "geog_in" values will be from geography config.
        If the requested geography is not custom, "level" and "geog_in" values will be from request config.

        Returns:
            dict: dict containing, at least, "level" and "geog_in"
        """
        if self._geographic_level_is_custom():
            return self.geography_config[self.request_config["geography"][self.geography_index]["level"]] # will also have "file" and "groupby" attribute
        else:
            return self.request_config["geography"][self.geography_index]
        
    def _get_geog_in(self, requested_level_config: dict) -> dict:
        """
        Get the information required to filter the census api response to only the needed rows.

        Args:
            requested_level_config (dict): dict from the geographies config that the request file asks for

        Raises:
            ValueError: If no match exists between request geography and geographies file.

        Returns:
            dict: geog_in details incuding geog, url_params, and file path
        """
        # go through each possible geog_in dict in the geog_in list of dicts
        for geog_in in requested_level_config["geog_in"]:
            # return the geog_in dict that matches the requested geog_in
            if geog_in["geog"] == self.request_config["geography"][self.geography_index]["geog_in"]:
                return geog_in
        raise ValueError('Could not find request geography in geographies.json')

    def _get_geog_file_path(self, base_geography_level: str, base_geography_geog_in: str) -> str:
        """
        Gets the file path for the geography being used for filtering out the rows in the API request response dataframe.

        Args:
            base_geography_level (str): geographic level of api response
            base_geography_geog_in (str): geog_in filter to be used

        Raises:
            ValueError: If file path or geography cannot be found

        Returns:
            str: File Path to a csv listing geoids to include
        """
        for geog_in in self.geography_config[base_geography_level]['geog_in']:
            if geog_in["geog"] == base_geography_geog_in:
                return geog_in["file"]
        raise ValueError('Could not find request geography in geographies.json')

    def _get_field_item_src(self, field:dict) -> dict:
        """
        Retrieves a field in fields source based on source_id.

        Args:
            field (dict): an element of fields list. Containing source_id key.

        Raises:
            ValueError: If a field in fields source_id does not match any source in sources id

        Returns:
            dict: Source information (id, src, config)
        """
        for src in self.request_config["sources"]:
            if src["id"] == field["source_id"]:
                return src
        raise ValueError(f'Could not find field source id: {field["source_id"]} in request sources json object array.')
    
    def _make_param(self) -> str|None:
        # go through each possible level in the geography configuration file
        for level in self.geography_config.keys(): # level = 'place', 'tract', 'zip', 'council_district', etc.
            # if the current request geography level matches...
            if self._get_geographic_level()['level'] == level:
                # get requested level configuration
                requested_level_config: dict = self.geography_config[level]
                geog_in = self._get_geog_in(requested_level_config)
                # check to make sure there are url params, if not, is a national dataset and is pulling all records for the geography.
                # ... zip codes does this
                if geog_in.get("url_params"):
                    return f'&{fill_param(requested_level_config["url"], geog_in["url_params"])}'
                else:
                    return f'&{requested_level_config["url"]}'

    def _generate_api_calls(self):
        """
        Generates API Calls

        Yields:
            str: url to call
        """
        self.master_field_list = ['geoid']
        for field_item in self._get_request_fields():
            field_list = ['NAME']
            for field_number in field_item['fields']:
                for field_type in self.field_output_types:
                    field_list.append(make_census_field_name(field_item['table'], field_number, field_type = field_type))
            self.master_field_list.extend(field_list[1:])
            field_list_str = ','.join(field_list)
            src = self._get_field_item_src(field_item)
            if src["src"] == 'acs':
                yield f"https://api.census.gov/data/{src['config']['year']}/acs/acs{src['config']['version']}?get={field_list_str}{self._make_param()}"
        
    def _get_and_load_data(self):
        """
        Gets Data from API

        Raises:
            ValueError: Alerts if no data was returned

        Yields:
            Dataframe: Loaded Data
        """
        for url in self._generate_api_calls():
            response = requests.get(url)
            if response.status_code == 200 and response.text[:5] != 'error':
                yield pd.DataFrame.from_records(response.json()[1:], columns=response.json()[0])
            else:
                raise ValueError(f'Bad response. Please check number of parameters and url. Url attempted: {url}')

    def _calc_aggregate(self, in_df: pd.DataFrame) -> pd.DataFrame:
        if self._geographic_level_is_custom():
            portion_field = self.request_config["geography"][self.geography_index]["portion_field"]
            base_geography = self._get_geographic_level()
            app_df = pd.read_csv(base_geography['file'])
            app_df.geoid = app_df.geoid.astype(str)
            
            df_master = app_df.set_index('geoid').join(in_df.set_index('geoid'), on='geoid', how='inner')
            
            # if column name starts with a capital B or C (it is ACS Data column)
            data_cols = [col for col in df_master.columns if col[0] in ['B','C'] ]
            # if acs data column ends in M, it is a margin of error column
            data_cols_m = [col for col in df_master.columns if col[-1] == 'M']
            for col in data_cols:
                if col in data_cols_m:
                    df_master[col] = (df_master[col].astype(float)**2) * df_master[portion_field]
                else:
                    df_master[col] = df_master[col].astype(float) * df_master[portion_field]
            # Group by
            # for neighborhoods, base_geography['groupby'] = nh_id
            data_cols.append(base_geography['groupby'])
            data_cols.append(base_geography['geog_type_field'])
            df_master.reset_index(drop=True, inplace=True)
            df_calc = df_master[data_cols]
            
            df_calc = df_calc.groupby([base_geography['geog_type_field'], base_geography['groupby']]).sum()
            
            for col in data_cols_m:
                df_calc[col] = np.sqrt(df_calc[col].astype(float))
            df_calc.reset_index(drop=False, inplace=True)
            return df_calc
        else:
            return in_df
        
    def _calc_do_custom_calcs(self, in_df: pd.DataFrame) -> pd.DataFrame:
        if self.request_config.get('calc'):
            for calculation in self.request_config['calc']:
                if calculation['math'].lower() == 'sum':
                    field_lists = {}
                    for field in calculation['from_fields']:
                        # for each field output type (E, M, or E and M)
                        for field_type in self.field_output_types.upper():
                            if field_lists.get(field_type) is None:
                                field_lists[field_type] = []
                            field_lists[field_type].append(make_census_field_name(field['table'], field['field'], field_type = field_type))
                    # Estimate
                    if 'E' in self.field_output_types.upper():
                        in_df[calculation['field']] = in_df[field_lists['E']].astype(float).sum(axis=1)
                    # MOE
                    if 'M' in self.field_output_types.upper():
                        to_sum_sq_error = []
                        for moe in field_lists['M']:
                            in_df[moe + "_sq"] = np.square(in_df[moe].astype(float))
                            to_sum_sq_error.append(moe + "_sq")
                        in_df[calculation['field']+'_MOE'] = np.sqrt(in_df[to_sum_sq_error].sum(axis=1))
                        in_df.drop(columns=to_sum_sq_error, inplace=True)
            return in_df
        else:
            return in_df
    def _split_df_by_agg_field(self, gtf, agg_list, flat_df:pd.DataFrame):
            if agg_list == "all":
                derived_agg_list = list(flat_df[gtf].unique())
                return self._split_df_by_agg_field(gtf, derived_agg_list, flat_df)
            else:
                flat_dfs = {}
                for agg in agg_list:
                    flat_dfs.update({agg : flat_df[flat_df[gtf] == agg]})
                return flat_dfs

    def process(self, out_folder_path='outputs'):
        self._initialize_configs()
        self.geography_index = 0 
        while self.geography_index < len(self.request_config["geography"]):
            base_geography = self._get_geographic_level()
            # List to hold filtered datasets
            df_list=[]
            for df in self._get_and_load_data():
                # generate geoid using base (non-custom) level
                df['geoid'] = generate_geoids(df, base_geography['level'])
                # load list of base geometries to filter
                geom_df = pd.read_csv(self._get_geog_file_path(base_geography['level'], base_geography['geog_in']))
                geom_df.geoid = geom_df.geoid.astype(str)
                # inner join on geoid to filter request results
                df_joined = geom_df.set_index('geoid').join(df.set_index('geoid'), on='geoid', how='inner')
                df_list.append(df_joined)
            # flatten filtered dfs
            igen = (i for i in range(len(df_list)*2))
            flat_df = reduce(lambda x,y: pd.merge(x, y, on='geoid', how='inner', suffixes=(str(next(igen)), str(next(igen)))), df_list)
            flat_df = flat_df.reset_index()[self.master_field_list]
            # aggregate and do custom calcs (if needed)
            flat_df = self._calc_aggregate(flat_df)
            flat_df = self._calc_do_custom_calcs(flat_df)
            
            # drop non_custom columns if applicable
            flat_dfs = {self.request_config["geography"][self.geography_index]['level'] : flat_df}
            if self._geographic_level_is_custom():
                agg_list = self.request_config["geography"][self.geography_index]["aggs"]
                flat_dfs = self._split_df_by_agg_field(base_geography['geog_type_field'], agg_list, flat_df)
            
            for gg, f_df in flat_dfs.items():
                if self._geographic_level_is_custom():
                    f_df = f_df.drop(columns=[base_geography['geog_type_field']])
                if self.calc_columns_only:
                    if self._geographic_level_is_custom():
                        f_df = f_df.drop(columns=[col for col in self.master_field_list if col in flat_df.columns])
                    else:
                        cc = [col for col in self.master_field_list if col in f_df.columns]
                        cc.remove('geoid')
                        f_df = f_df.drop(columns=cc)
                out_geog = self.request_config["geography"][self.geography_index]
                if self._geographic_level_is_custom():
                    out_path = os.path.join(out_folder_path, f"{os.path.basename(self.request_config_path).split('.')[0]}_{out_geog['geog_in']}_{out_geog['portion_field'].split('_')[0]}_{gg}")
                else:    
                    out_path = os.path.join(out_folder_path, f"{os.path.basename(self.request_config_path).split('.')[0]}_{out_geog['geog_in']}_census_{gg}")
                f_df.to_csv(f"{out_path}.csv", index=False)

            # iterate
            self.geography_index += 1