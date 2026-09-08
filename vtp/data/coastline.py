"""
Loads land polygons and port locations for a maritime domain.
Uses Natural Earth (via geopandas) for coastline, and a ports CSV
(lon, lat, name columns) for port nodes.
"""
import geopandas as gpd
from shapely.geometry import box
import pandas as pd


def load_land_polygons(bounds, natural_earth_path=None, resolution='10m'):
    """
    bounds: (min_lon, min_lat, max_lon, max_lat)
    natural_earth_path: path to a local Natural Earth land shapefile
        (ne_10m_land.shp). Rivanna compute nodes have no general internet
        egress, so stage this ahead of time on a login node.
    Returns: list of shapely Polygons clipped to bounds.
    """
    path = natural_earth_path or "./data/ne_10m_land/ne_10m_land.shp"
    land = gpd.read_file(path)
    domain = box(*bounds)
    clipped = gpd.clip(land, domain)
    polygons = []
    for geom in clipped.geometry:
        if geom.geom_type == 'Polygon':
            polygons.append(geom)
        elif geom.geom_type == 'MultiPolygon':
            polygons.extend(list(geom.geoms))
    return polygons


def load_ports(ports_csv_path, bounds):
    """
    ports_csv_path: CSV with columns [name, lon, lat].
    Returns: list of (lon, lat) tuples within bounds.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    df = pd.read_csv(ports_csv_path)
    mask = (
        (df['lon'] >= min_lon) & (df['lon'] <= max_lon) &
        (df['lat'] >= min_lat) & (df['lat'] <= max_lat)
    )
    df = df[mask]
    return list(zip(df['lon'], df['lat']))


# South China Sea -- shelved for now, kept in case you return to it.
SCS_BOUNDS = (99.0, -3.0, 122.0, 24.0)

# Danish home waters -- North Sea, Skagerrak/Kattegat, and the western
# Baltic around Denmark. Matches the coverage of DMA's bulk AIS archive.
DMA_BOUNDS = (7.0, 53.5, 16.0, 58.5)
