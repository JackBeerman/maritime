"""
Builds an adaptive triangular mesh over a maritime domain.
Denser near coastlines and ports, coarser in open water.
"""
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, shape
from shapely.prepared import prep


def sample_domain_points(bounds, land_polygons, ports, n_open_water=3000,
                          n_coastal=4000, coastal_band_km=50):
    """
    bounds: (min_lon, min_lat, max_lon, max_lat)
    land_polygons: list of shapely Polygons (coastline / landmass)
    ports: list of (lon, lat) port locations
    Returns: (N, 2) array of lon/lat mesh node coordinates.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    prepared_land = [prep(p) for p in land_polygons]

    def is_land(lon, lat):
        pt = Point(lon, lat)
        return any(p.contains(pt) for p in prepared_land)

    # 1. Uniform background sampling (open water density)
    rng = np.random.default_rng(0)
    open_pts = rng.uniform(
        low=[min_lon, min_lat], high=[max_lon, max_lat], size=(n_open_water, 2)
    )
    open_pts = np.array([pt for pt in open_pts if not is_land(*pt)])

    # 2. Coastal band densification: sample near land polygon boundaries
    coastal_pts = []
    deg_per_km = 1.0 / 111.0
    band = coastal_band_km * deg_per_km
    for poly in land_polygons:
        boundary = poly.exterior
        n = int(n_coastal / max(len(land_polygons), 1))
        dists = rng.uniform(0, boundary.length, size=n)
        for d in dists:
            pt = boundary.interpolate(d)
            # jitter perpendicular-ish into water by a random offset within the band
            jitter = rng.uniform(-band, band, size=2)
            cand = (pt.x + jitter[0], pt.y + jitter[1])
            if min_lon <= cand[0] <= max_lon and min_lat <= cand[1] <= max_lat:
                coastal_pts.append(cand)
    coastal_pts = np.array(coastal_pts) if coastal_pts else np.empty((0, 2))

    # 3. Force port locations as exact nodes, plus a dense ring around each
    port_pts = []
    for (plon, plat) in ports:
        port_pts.append((plon, plat))
        ring = rng.normal(scale=0.05, size=(30, 2)) + np.array([plon, plat])
        port_pts.extend(ring.tolist())
    port_pts = np.array(port_pts)

    all_pts = np.vstack([p for p in [open_pts, coastal_pts, port_pts] if len(p) > 0])
    return all_pts, [is_land(*pt) for pt in all_pts]


def build_mesh(points, is_land_flags, ports, port_radius_deg=0.1):
    """
    Triangulates points and returns node features + edge_index (undirected, deduped).
    """
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edges.add((min(a, b), max(a, b)))
    edge_index = np.array(list(edges)).T  # (2, E)

    is_port = np.zeros(len(points), dtype=bool)
    for (plon, plat) in ports:
        d = np.sqrt((points[:, 0] - plon) ** 2 + (points[:, 1] - plat) ** 2)
        is_port |= d < port_radius_deg

    node_features = np.stack([
        points[:, 0],                      # lon
        points[:, 1],                      # lat
        np.array(is_land_flags, dtype=float),
        is_port.astype(float),
    ], axis=1)

    return node_features, edge_index, tri
