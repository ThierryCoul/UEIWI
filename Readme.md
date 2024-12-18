## **Delineating Cambodian Cities and Assessing Their Sustainable Growth with the Inclusive Wealth Index**

Codes for the manuscript \`Delineating Cambodian Cities and Assessing Their Sustainable Growth with the Inclusive Wealth Index'.

The materials in this repository allow users to visualize the code used to create parts of the data of the manuscript.

If you find meaningful errors in the code or have questions or suggestions, please contact Thierry Yerema Coulibaly at [yerema.coul\@gmail.com](mailto:yerema.coul@gmail.com){.email}

## Organization of repository

-   **Upsample_MODIS_Random_Forest.js**: File describing the pipeline used to upsample the MODIS 500m resolution to 30m resolution based on Landsat imageries with Google Earth Engine.
-   **Random_Forest_of_Landsat.js**: File describing the pipeline used to train Landsata images to rasters with water, urban and open areas with Google Earth Engine.
-   **Atlas_of_cities.ipynb**: File describing the pipeline used to delineate cities based on raster values with ArcPy on Python.
-   **Cities Boundaries**: Folder containing the shapefile of the cities created.
-   **IWI_area_data.csv**: Statistics used for the plots.

## Python packages required

-   **arcpy**

Our output can be accessed here on the page: <iframe src="https://thierrycoul.github.io/CambodiaCities.github.io/Urban_extent.html" width="100%" height="600" style="border:none;"></iframe>
