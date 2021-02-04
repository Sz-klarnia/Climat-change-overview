# Climat-change-overview---rising-temperatures-analysis-and-prediction

## Project info:
Based on various datasets describing various aspects of Climate Change, mainly CO2 emissions, concentration levels and temperatures I aim to create as accurate climate predictions as I can. 

### Analyses
Analyses are avaliable in global_warming_analyses notebook. 

#### CO2 levels
All analyses show dynamic rise in CO2 and other greenhouse gasses emissions. These changes affect atmospheric concentration levels directly, as they are also quickly rising. Currently, greenhouse gasses concentration levels are higher than in any point in the last 800 thousand years, and probably for much, much longer. During Holocene and Pleistocene these levels fluctuated between 180 ppm in Glacial periods and 280-300 ppm in interglacials. 
#### GDP and emissions
It is also clearly visible, that GDP rise globally is strongly correlated with emissions rise - it means that current development pattern is strongly tied to increased emissions, and humanity must finde other ways to develop, as this pattern is not sustainable
#### Land Use and deforestation
Land Use also contributes to the rise of emissions, mainly from farming, deforestation etc. Globally percentage of forests shrinked by 2% in the last 25 years, but in some countries this change is much more dramatical. Developing countries with large areas of tropical rainforests, such as Brazil or Indonesia are rapidly reducing their forest cover, changing one of the biggest ecosystems

### Modelling
Models are avaliable in Modelling Global Warming file. 

I created models for 3 scenarios of global warming - no change in emission trends, net zero emissions by 2050, energetics decarbonization by 2050.
Only in second of these scenarios the temperature levels do not exceed the 2 celcius threshold, and even in this one they exceed 1.5 celcius threshold. In both others, we exceed 2 celcius threshold before 2050, and in the firs one by 2095 the temperatures rise dramatically, by more than 4,5 degrees.
These analyses, flawed as they currently are show how real and close the threat of global warming is.

### Next plans

In the future I want to add more detailed analyses of co2 emissions to extract more info on how various countries contribute to climate change and add more detailed visualizations. I also want to add more features to my predictions and analyses, such as information on albedo changes and permafrost thawing, and create more scenarios


### Data sources
Filename: GlobalLandTemperaturesByCity.csv
source: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
File not in the repository because of it's size

Filename: Greenhouse_Cooncentration.csv
sources: https://www.eea.europa.eu/data-and-maps/daviz/atmospheric-concentration-of-carbon-dioxide-5#tab-chart_5_filters=%7B%22rowFilters%22%3A%7B%7D%3B%22columnFilters%22%3A%7B%22pre_config_polutant%22%3A%5B%22CH4%20(ppb)%22%5D%7D%7D

Filename: owid_co2_data.csv
source: https://github.com/owid/co2-data

Filename: FAOSTAT_data_1-27-2021.csv
source: http://www.fao.org/faostat/en/#data/GL

Filename: API_AG.LND.FRST.ZS_DS2_en_csv_v2_1926709.csv
Source: https://data.worldbank.org/indicator/AG.LND.FRST.ZS
