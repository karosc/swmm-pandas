## Input File Interface Progress

| Section       | Parser             | Testing            | Structure Type                     | TODO                                                  |
|---------------|--------------------|--------------------|------------------------------------|-------------------------------------------------------|
| TITLE         | :heavy_check_mark: | :heavy_check_mark: | String                             |                                                       |
| OPTIONS       | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| REPORT        | :heavy_check_mark: | :heavy_check_mark: | Custom Object                      |                                                       |
| EVENT         | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| FILES         | :x:                | :x:                | String                             | build better interface than text interface            |
| RAINGAGES     | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| EVAPORATION   | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          | better column names                                   |
| TEMPERATURE   | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| ADJUSTMENTS   | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| SUBCATCHMENTS | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| SUBAREAS      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| INFILTRATION  | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          | better column names                                   |
| LID_CONTROLS  | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| LID_USAGE     | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| AQUIFERS      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| GROUNDWATER   | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| GWF           | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| SNOWPACKS     | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| JUNCTIONS     | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| OUTFALLS      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| DIVIDERS      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| STORAGE       | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| CONDUITS      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| PUMPS         | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| ORIFICES      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| WEIRS         | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| OUTLETS       | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| XSECTIONS     | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| TRANSECTS     | :x:                | :x:                | String                             |                                                       |
| STREETS       | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| INLETS        | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| INLET_USAGE   | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| LOSSES        | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| CONTROLS      | :x:                | :x:                | String                             | build better interface than text interface            |
| POLLUTANTS    | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| LANDUSES      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| COVERAGES     | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          | Support parsing multiple coverages on a single line   |
| LOADINGS      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          | Support parsing multiple loadings on a single line    |
| BUILDUP       | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          | Does it make sense to index by landuse and pollutant? |
| WASHOFF       | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| TREATMENT     | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| INFLOWS       | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| DWF           | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| RDII          | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| HYDROGRAPHS   | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| CURVES        | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| TIMESERIES    | :heavy_check_mark: | :heavy_check_mark: | Hashmap of Dataframes and Objects  | Better output string formatting.                      |
| PATTERNS      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| MAP           | :x:                | :x:                | String                             | Build a better data structure for this.               |
| POLYGONS      | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          | Add coordinate index                                  |
| COORDINATES   | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
| VERTICIES     | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          | Add coordinate index                                  |
| LABELS        | :heavy_check_mark: | :x:                | Dataframe                          |                                                       |
| SYMBOLS       | :heavy_check_mark: | :x:                | Dataframe                          |                                                       |
| BACKDROP      | :x:                | :x:                | String                             |                                                       |
| PROFILE       | :x:                | :x:                | String                             |                                                       |
| TAGS          | :heavy_check_mark: | :heavy_check_mark: | Dataframe                          |                                                       |
