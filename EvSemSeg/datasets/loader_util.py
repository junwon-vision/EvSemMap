# Available Remap: V3
import pandas as pd
import torchvision.transforms as transforms

def inv_transform(img) :
    invTrans = transforms.Compose([ transforms.Normalize(mean = [0., 0., 0.],
                                                        std = [1/0.229, 1/0.224, 1/0.225]),
                                    transforms.Normalize(mean = [-103.939/255, -116.779/255, -123.68/255],
                                                        std = [1., 1., 1.])
                                    ])
    return invTrans(img)

### RELLIS-3D Original #############################################################################################
RELLIS_ORG_PALETTE = [ # RELLIS-3D has 20 Classes 
    [0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153],
    [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64],
    [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204],
    [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138],
]
RELLIS_ORG_ID = [
    0, 1, 3, 4, 5,
    6, 7, 8, 9, 10,
    12, 15, 17, 18, 19,
    23, 27, 31, 33, 34
]

RELLIS_ORG_ID_COMPACT20 = [
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9,
    10, 11, 12, 13, 14,
    15, 16, 17, 18, 19
]

rellis_orgid2color = {id: color for id, color in list(zip(RELLIS_ORG_ID, RELLIS_ORG_PALETTE))}
rellis_orgid_compact20_to_color = {id: color for id, color in list(zip(RELLIS_ORG_ID_COMPACT20, RELLIS_ORG_PALETTE))}
rellis_orgid2_orgid_compact20 = {id1: id2 for id1, id2 in list(zip(RELLIS_ORG_ID, RELLIS_ORG_ID_COMPACT20))}

### RemapV3
# V3 Remapped ID to V3 Color
remapV3color = pd.read_csv('/kjyoung/v3_v1_colormap_paper_unified.txt', sep="\t", header=None)
remapV3color.columns = ["new_id", "new_name", "R", "G", "B"]
V3_newID2color = {id: [r, g, b] for (id, r, g, b) in zip(remapV3color.new_id, remapV3color.R, remapV3color.G, remapV3color.B)}

# V3 Rellis&RUGD ID to V3 ID
remapV3 = pd.read_csv('/kjyoung/v3_v1.txt', sep="\t", header=None)
remapV3.columns = ["org_id", "org_name", "new_id"]
V3_rugd_rellis_orgID2remapID = {id: newID for (id, newID) in zip(remapV3.org_id, remapV3.new_id)}


RemapV3 = {
    'n_class': 9,
    'rellis_orgID2remapID': V3_rugd_rellis_orgID2remapID,
    'rugd_orgID2remapID': V3_rugd_rellis_orgID2remapID,
    'id2color': V3_newID2color,
}

RemapOBJ = {
    3: RemapV3,
}