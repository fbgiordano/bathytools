import logging
import warnings
from pathlib import Path
from bathytools.utilities.relative_paths import read_path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
from bitsea.basins.region import Polygon
from bitsea.components.component_mask import ComponentMask

#from bathytools import geoarrays
from bathytools.actions import SimpleAction
from bathytools.output_appendix import OutputAppendix
from bathytools.utilities.points import Point
from bathytools.utilities.points import Segment
from bathytools.utilities.relative_paths import read_path


LOGGER = logging.getLogger(__name__)

class IncludeExtBathy(SimpleAction):
    """
    Modifies the values of the bathymetry dataset by including data sources
    other than EMODnet ones. Given the diverse origins, spatial coverages and
    formats of these data, ad hoc solutions are present throughout this action,
    as well as a relatively large amount of config inputs.
    Including further datasets would probably require the user to tweak the
    existing code if not outright coming up with their own solutions. 

    This action accepts the following arguments:
      - input_files (dict): A dictionary of strings, pointing to the datasets
          paths.
      - input_crs (dict): A dictionary of CRS codes, for the datasets that need
        to be transformed to a cartesian (lat-lon) coordinate system.
      - window (dict): coordinates of boxes bounding the dataset's domain.
      - weights (dict): weights representing the user-defined confidence
          of the different datasets when averaging them with the
          EMODnet (='orig') bathymetry.
    """
    def __init__(
        self,
        name: str,
        description: str,
        output_appendix: OutputAppendix,
        input_files: dict[Path, Path, Path, Path, Path],
        window: dict[dict, dict, dict, dict, dict],
        input_crs: dict[str, str, str, str, str],
        weights: dict[float, float, float, float, float, float],
    ):
        super().__init__(name, description, output_appendix=output_appendix)

        self._input_files = input_files
        self._window = window
        self._crss = input_crs
        self._weights = weights
    #
    # reoccurring functions
    def __windower__(self, bathymetry, fname):
        """
        Given a window (min-max lat and lon), selects the grid -centers and -borders coordinates
        as well as 4× resolution corresponding ones (used in the __unstruct2struct__ method)
        """
        xc = bathymetry.longitude.sel(longitude = slice(self._window[fname]['min_lon'], self._window[fname]['max_lon'])).values
        yc = bathymetry.latitude.sel(latitude = slice(self._window[fname]['min_lat'], self._window[fname]['max_lat'])).values
        xg = np.zeros(len(xc)+1)
        xg[0] = xc[0] - 0.5*np.diff(xc).mean()
        xg[1:] = xc + 0.5*np.diff(xc).mean()
        yg = np.zeros(len(yc)+1)
        yg[0] = yc[0] - 0.5*np.diff(yc).mean()
        yg[1:] = yc + 0.5*np.diff(yc).mean()
        xg4 = np.linspace(xg[0], xg[-1], 4*(len(xg)-1)+1)
        xc4 = 0.5 * (xg4[1:] + xg4[:-1])
        yg4 = np.linspace(yg[0], yg[-1], 4*(len(yg)-1)+1)
        yc4 = 0.5 * (yg4[1:] + yg4[:-1])
        return xc, yc, xg, yg, xc4, yc4, xg4, yg4
    #
    def __unstruct2struct__(self, bathymetry, fname):
        """
        Loads and interpolates, on a finer (4×) grid, unstructured bathymetry datasets
        """
        _, _, _, _, xc4, yc4, xg4, yg4 = self.__windower__(bathymetry, fname)
        df = pd.read_csv(read_path(self._input_files[fname]))
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude'], df['Depth']),
            crs=self._crss[fname]).to_crs(crs='EPSG:4326')['geometry']
        sum_z = np.histogram2d(gdf.x, gdf.y, weights=gdf.z, density=False, bins=(xg4, yg4))[0]
        num_z = np.histogram2d(gdf.x, gdf.y, density=False, bins=(xg4, yg4))[0]
        intdf = np.where(num_z > 0, sum_z/num_z, np.nan)
        return intdf
    ###
    def __call__(self, bathymetry):
        # -defining the domain coordinates arrays
        LOGGER.info("Loading and Interpolating bathymetry datasets")
        xc0 = bathymetry.longitude.values
        yc0 = bathymetry.latitude.values
        xg0 = np.zeros(len(xc0)+1)
        xg0[0] = xc0[0] - 0.5*np.diff(xc0).mean()
        xg0[1:] = xc0 + 0.5*np.diff(xc0).mean()
        yg0 = np.zeros(len(yc0)+1)
        yg0[0] = yc0[0] - 0.5*np.diff(yc0).mean()
        yg0[1:] = yc0 + 0.5*np.diff(yc0).mean()
        #### -file1
        fl = 'file1'
        xc, yc, xg, yg, xc4, yc4, xg4, yg4 = self.__windower__(bathymetry, fl)
        #
        intdf1 = self.__unstruct2struct__(bathymetry, fl)
        # -interpolate back on the original grid
        da1 = xr.DataArray(intdf1, coords = {'x': xc4, 'y': yc4}).coarsen({'x': 4, 'y': 4}).mean()
        da1 = da1.interp(x = xc0, y = yc0, method = 'linear', kwargs={"fill_value": np.nan})
        #### -file2
        fl = 'file2'
        xc, yc, xg, yg, xc4, yc4, xg4, yg4 = self.__windower__(bathymetry, fl)
        # ! here we limit the structured dataset to the water points of the previous one;
        #   this is because we trust more the former, due to its denser sampling !
        intdf2 = self.__unstruct2struct__(bathymetry, fl) * np.where(intdf1 == intdf1, 1, np.nan)
        # -interpolate back on the original grid
        da2 = xr.DataArray(intdf2, coords = {'x': xc4, 'y': yc4}).coarsen({'x': 4, 'y': 4}).mean()
        da2 = da2.interp(x = xc0, y = yc0, method = 'linear', kwargs={"fill_value": np.nan})
        #### -file3
        fl = 'file3'
        xc, yc, xg, yg, xc4, yc4, xg4, yg4 = self.__windower__(bathymetry, fl)
        #
        intdf3 = self.__unstruct2struct__(bathymetry, fl)
        # -interpolate back on the original grid
        da3 = xr.DataArray(intdf3, coords = {'x': xc4, 'y': yc4}).coarsen({'x': 4, 'y': 4}).mean()
        da3 = da3.interp(x = xc0, y = yc0, method = 'linear', kwargs={"fill_value": np.nan})
        #### -file4
        fl = 'file4'
        xc, yc, xg, yg, xc4, yc4, xg4, yg4 = self.__windower__(bathymetry, fl)
        #
        intdf4 = self.__unstruct2struct__(bathymetry, fl)
        # -interpolate back on the original grid
        da4 = xr.DataArray(intdf4, coords = {'x': xc4, 'y': yc4}).coarsen({'x': 4, 'y': 4}).mean()
        da4 = da4.interp(x = xc0, y = yc0, method = 'linear', kwargs={"fill_value": np.nan})
        #### -file5
        fl = 'file5'
        #
        df = xr.open_dataarray(read_path(self._input_files[fl])).T
        # -interpolate back on the original grid
        da5 = -df.interp(lon = xc0, lat = yc0, method = 'nearest')
        da5 = xr.where(da5 < 0., da5, 0.)
        #### -weighted average of the different datasets
        bathys_list = [da1, da2, da3, da4, da5, bathymetry.elevation]
        keys_list = [f'file{i}' for i in range(1,6)] + ['orig']
        bathys_list = [self._weights[fl] * bathys_list[fli] for fli, fl in enumerate(keys_list)]
        weights_list = [self._weights[fl] * np.where(np.isnan(bathys_list[fli]), np.nan, 1) for fli, fl in enumerate(keys_list)]
        avg_bathy = np.nansum(bathys_list, axis = 0) / np.nansum(weights_list, axis = 0)
        #
        bathymetry.elevation.loc[:,:] = avg_bathy
        return bathymetry

#

class FixVolumesLagoons(SimpleAction):
    """
    
    """
    def __init__(
        self,
        name: str,
        description: str,
        output_appendix: OutputAppendix,
        polygons: Path,
        threshold: float,
        min_depth: float,
    ):
        super().__init__(name, description, output_appendix=output_appendix)

        self._polygons = polygons
        self._threshold = threshold
        self._min_depth = min_depth
    #
    def __geo_areas__(self, bathymetry):
        R0 = 6.371e6
        xc0 = bathymetry.longitude.values
        yc0 = bathymetry.latitude.values
        xg0 = np.zeros(len(xc0)+1)
        xg0[0] = xc0[0] - 0.5*np.diff(xc0).mean()
        xg0[1:] = xc0 + 0.5*np.diff(xc0).mean()
        yg0 = np.zeros(len(yc0)+1)
        yg0[0] = yc0[0] - 0.5*np.diff(yc0).mean()
        yg0[1:] = yc0 + 0.5*np.diff(yc0).mean()
        Xg0, _ = np.meshgrid(xg0, yc0)
        _, Yg0 = np.meshgrid(xc0, yg0)
        #
        dXg = (np.diff(Xg0, axis = 1).T * R0 * np.pi/180 * np.cos(yc0 * np.pi/180)).T
        dYg = np.diff(Yg0, axis = 0) * R0 * np.pi/180
        dA = (dXg * dYg).T
        return dA
    #
    def __call__(
        self,
        bathymetry,
    ):
        LOGGER.info("Fixing lagoons bathymetries and conserving their volume")
        with open(read_path(self._polygons), "r") as f:
            available_polys = Polygon.read_WKT_file(f)
        keys_polys = list(available_polys.keys())
        #
        for key_p in keys_polys:
            poly = available_polys[key_p]
            window_poly = poly.is_inside(
                lon=bathymetry.longitude.values,
                lat=bathymetry.latitude.values[:, np.newaxis],
            ).T
            dA = self.__geo_areas__(bathymetry) * window_poly
            #
            lagoon = bathymetry.elevation.values * window_poly
            # -compute initial lagoon volume
            ini_vol = np.abs(lagoon * dA)[(lagoon != 0) * (~np.isnan(lagoon))].sum()
            # -fix minimum depth
            lagoon = np.where(lagoon > -self._threshold[key_p], 0.,
                np.min([-self._min_depth[key_p] * np.ones_like(lagoon), lagoon], axis=0))
            # -set to land isolated ponds
            water_cells = lagoon < 0.0
            components = ComponentMask(water_cells)
            sea_cells = components.get_component(components.get_biggest_component())
            outside_main_component = np.logical_not(sea_cells)
            lagoon[outside_main_component] = 0.
            # -compute final volume and variation
            fin_vol = np.abs(lagoon * dA)[(lagoon != 0) * (~np.isnan(lagoon))].sum()
            d_vol = fin_vol - ini_vol
            # -spread volume variation over the remaining water points
            lagoon[(lagoon != 0) * (lagoon == lagoon)] += d_vol/dA[(lagoon != 0) * (lagoon == lagoon)] / len(lagoon[(lagoon != 0) * (lagoon == lagoon)])
            bathymetry.elevation.values[window_poly] = lagoon[window_poly]
            # - check volumes; possibly to be moved to a logging call
            LOGGER.info(f'Initial {key_p} volume: V₀ = {np.abs(ini_vol)*1e-9:.3f} km³')
            LOGGER.info(f'Final {key_p} volume: V₁ = {np.abs(fin_vol)*1e-9:.3f} km³')
            LOGGER.info(f'Variation in volume: ΔV = {d_vol*1e-9:.3f} km³')
        #
        return bathymetry
