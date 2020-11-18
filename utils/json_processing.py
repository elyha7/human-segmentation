from PIL import Image
import io
import json
import zlib
import base64
import numpy as np
import cv2
import shapely
from shapely.geometry import Polygon
def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype('uint8')
    return mask

def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')
def get_masks(filename):
    with open(filename) as f:
        info = json.load(f)
    background = np.zeros((info['size']['height'],info['size']['width']),dtype='uint8')
    for obj in info['objects']:
        if obj['bitmap'] is None:
            if obj['classTitle'] == 'neutral':
                continue
            _ = create_mask(background,obj)
        else:
            position = obj['bitmap']['origin']
            msk = base64_2_mask(obj['bitmap']['data'])
            background[position[1]:position[1]+msk.shape[0],position[0]:position[0]+msk.shape[1]] = msk
    return background*1.0/background.max()
def mask_for_polygons(background,polygons):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    img_mask = background
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask
def mask_to_polygons(mask, epsilon=10., min_area=10.):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    image, contours, hierarchy = cv2.findContours(mask,
                                  cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiPolygon()
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    all_polygons = MultiPolygon(all_polygons)

    return all_polygons
def create_mask(background,obj):
    masks = []
    poly = Polygon(obj['points']['exterior'],holes=obj['points']['interior'])
    mask = mask_for_polygons(background,[poly])
    return mask