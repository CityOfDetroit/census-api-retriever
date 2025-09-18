## Overview
This folder should contain any custom apportionment csvs.

The csvs should contain a field for the geography type of each row (naming determined by user), a unique id to later group each row / geography type combination, a geoid for the census geography/segment denoted by each row, at least one portion field, denoting the portion of the entire census geography represented by that segment. There should be only 1 census geography type per csv, but each csv can contain multiple custom geog_types.