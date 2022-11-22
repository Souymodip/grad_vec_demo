import os
import config as cg
import scipy.spatial
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import ChargingBar
import scipy.optimize
from merge import lin_merge
from stroke import main as edgy
from vectorize import main as vec
import cv2
from collections import Counter


class FeatureKey:
    def __init__(self, spatial_radius=10, colour_radius=0.2, gradient_radius=None):
        self.values=[]
        self.values.append(spatial_radius)
        self.values.append(colour_radius)
        self.key=[0]*2+[1]*3
        self.has_gradients=gradient_radius is not None
        if gradient_radius is not None:
            self.values.append(gradient_radius)
            self.key+=[2]*6
        self.values=np.array(self.values,dtype=float)

    def make_scale(self):
        return self.values[self.key]


def flatten_features(img, grad=False):
    ix, iy=np.meshgrid(np.arange(img.shape[0]),np.arange(img.shape[1]), indexing="ij")

    if grad:
        gradients=[np.stack(np.gradient(img[:,:,i]),axis=2) for i in range(3)]
        features=np.concatenate([ix[:,:,None].astype(float),iy[:,:,None].astype(float),img,*gradients],axis=2)
    else:
        features=np.concatenate([ix[:,:,None].astype(float),iy[:,:,None].astype(float),img],axis=2)

    features=features.reshape((-1,features.shape[2]))

    return features


def unflatten_features(features, imsize, segment_indices=None):
    indices=features[:,:2].astype(int)
    values=features[:,2:5]

    img=np.zeros((*imsize,3),dtype=float)
    index_img=np.zeros(imsize,dtype=int) if segment_indices is not None else None

    for i in range(features.shape[0]):
        idx=indices[i,:]
        colour=values[i,:]
        img[idx[0],idx[1],:]=colour
        if segment_indices is not None:
            index_img[idx[0],idx[1]]=segment_indices[i]

    return img if segment_indices is None else (img, index_img)

def mean_shift(tree, scaled_features, query, threshold=.05):
    x=query
    while True:
        x_prev=x
        neighbour_indices=tree.query_ball_point(x,1.)
        neighbours=scaled_features[neighbour_indices,:]
        x=np.mean(neighbours,axis=0)

        step_len=np.linalg.norm(x_prev-x)
        if step_len<threshold:
            return x

def segment_features(features, feature_key):
    feature_scale=1./feature_key.make_scale()

    num_points=features.shape[0]

    # pre-scale the space so we can use a simple euclidean distance
    scaled_features=features*feature_scale[None,:]
    tree=scipy.spatial.KDTree(scaled_features)

    # run the actual mean shifts
    out_features=[]
    print("Iterating mean shift")
    with ChargingBar("Mean Shift: ",max=num_points) as cb:
        for i in range(num_points):
            out=mean_shift(tree,scaled_features,scaled_features[i,:])
            out_features.append(out)
            cb.next()
    out_features=np.stack(out_features,axis=0)

    print("Grouping output features")
    # here we group the output features to create segments with the help of the KDTree
    group_tree=scipy.spatial.KDTree(out_features)
    # this call returns pairs of points closer to each other than half the kernel radius
    # using a tree to tree version to avoid dealing with tons of pairs
    results=group_tree.query_ball_tree(group_tree,.5)

    print("Building mapping table")
    # we will now build this into a table to help us collapse point indices; each point will map to its lowest index
    # paired neighbour
    mapping_table=np.array([np.min(result) for result in results])

    # initialize the cluster index table to each pixel having its own cluster
    index_table=np.arange(num_points)

    print("Mapping")
    # now do transitive closure using the relation in mapping table to ensure all pxiels in segment have the same index
    ctr=0
    while True:
        old_index_table=index_table
        index_table=mapping_table[old_index_table]
        ctr+=1
        if np.all(old_index_table==index_table):
            break
    print("Mapping closure after {} iterations".format(ctr))

    # now let's relabel everything so indices are 1..n
    unique_indices, inverse_indices=np.unique(index_table,return_inverse=True)
    unique_index_table=np.arange(unique_indices.shape[0])[inverse_indices]

    return unique_index_table, out_features

def reconstruct_segment(positions, colours):
    x_init=np.random.rand(8)/255.

    def estimate_colours(x):
        g=x[:2]
        c0=x[2:5]
        c1=x[5:]

        t=np.sum(positions*g[None,:],axis=1)
        c_out=c0[None,:]+t[:,None]*c1[None,:]

        return c_out

    def colour_grad(x):
        g=x[:2]
        c1=x[5:]

        g_grad=c1[None,:,None]*positions[:,None,:]
        c0_grad=np.tile(np.eye(3)[None,:,:],[positions.shape[0],1,1])
        c1_grad=np.sum(positions*g[None,:],axis=1)[:,None,None]*np.eye(3)[None,:,:]
        grad=np.concatenate([g_grad, c0_grad, c1_grad],axis=2)

        return grad

    def fun(x):
        est_c=estimate_colours(x)
        dif=est_c-colours
        return np.concatenate([dif[:,c] for c in range(3)],axis=0)

    def grad(x):
        c_grad=colour_grad(x)
        return np.concatenate([c_grad[:,c,:] for c in range(3)], axis=0)

    res=scipy.optimize.least_squares(fun, x_init, jac=grad)
    est_colours=estimate_colours(res.x)

    return est_colours

def linear_reconstruct(flat_features, segments):
    positions=flat_features[:,:2]
    colours=flat_features[:,2:5]
    approx_colours=np.zeros(colours.shape)
    with ChargingBar("Gradient fitting: ",max=segments.max()+1) as cb:
        for idx in np.unique(segments):
            idx_vec=(segments==idx)
            approx_segment=reconstruct_segment(positions[idx_vec,:],colours[idx_vec,:])
            approx_colours[idx_vec,:]=approx_segment
            cb.next()
    return approx_colours

def segment_and_approx(shape, flat_segments, flat_features, debug):
    num_segments=np.max(flat_segments)+1
    np.random.seed(13)
    colormap=np.random.rand(num_segments,3)

    color_segments=colormap[flat_segments,:]
    print("flat sgements: {}, Image: {}".format(flat_segments.shape, shape))
    segment_img, segment_map=unflatten_features(
        np.concatenate([flat_features[:,:2],color_segments],axis=1),
        shape[:2],
        flat_segments
    )
    if debug:
        fig, ax = plt.subplots()
        ax.imshow(segment_img)
        ax.set_title("Segments :{}".format(num_segments))
        plt.show()
    rec_colours=linear_reconstruct(flat_features,flat_segments)
    rec_img=unflatten_features(
        np.concatenate([flat_features[:,:2],rec_colours],axis=1),
        shape[:2]
    )

    return segment_img, rec_img, segment_map

def np2pil(np_img):
    clamped=np.clip(np_img*255.,0.,255.)
    return clamped.astype(np.uint8)

def process_file(npimg):
    # dir_name, file_name=os.path.split(path)
    # file, ext=os.path.splitext(file_name)
    # print("Processing file {}".format(file_name))

    output_orig=os.path.join(cg.OUT_DIR, "out.png")

    npimg=npimg[:,:,:3].astype(float)/255.
    Image.fromarray(np2pil(npimg)).save(output_orig)
    return npimg


def reconstruct(shape, flat_segments, flat_features, file, ext, debug)  :
    output_seg=os.path.join(cg.OUT_DIR, file + "_segments" + ext)
    output_rec=os.path.join(cg.OUT_DIR, file + "_reconstruct" + ext)
    segment_img, rec_img, segment_map=segment_and_approx(shape, flat_segments, flat_features, debug=debug)
    Image.fromarray(np2pil(segment_img)).save(output_seg)
    Image.fromarray(np2pil(rec_img)).save(output_rec)


def segments_features(npimg, feature_key):
    flat_features = flatten_features(npimg, feature_key.has_gradients)
    flat_segments, out_features = segment_features(flat_features, feature_key)
    return flat_segments, flat_features

def clear_output(dirname):
    files_in_directory = os.listdir(dirname)
    filtered_files   = [file for file in files_in_directory if file.endswith(".png")]
    for file in filtered_files:
        path_to_file = os.path.join(dirname, file)
        os.remove(path_to_file)

def list_infiles(in_dir):
    files_in_directory = os.listdir(in_dir)
    filtered_files = [os.path.join(in_dir,file) for file in files_in_directory if file.endswith(".png")]
    return filtered_files


def build(npImg, feature_key, debug):
    npimg, file = process_file(npImg)
    npimg = npimg[-200:, -200:]
    segment, features = segments_features(npimg, feature_key)
    if debug:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(npimg)
        axs[1].imshow(segment.reshape(*npimg.shape[:2]))
        axs[0].set_title("Original")
        axs[1].set_title(f"Regions:{len(Counter(segment))}")
        plt.show()

    segment_ = lin_merge(img=npimg, flat_segment=segment, debug=False)
    if debug:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(segment.reshape(*npimg.shape[:2]))
        axs[1].imshow(segment_.reshape(*npimg.shape[:2]))
        axs[0].set_title(f"Regions:{len(Counter(segment))}")
        axs[1].set_title(f"Regions:{len(Counter(segment_))}")
        plt.show()
    segment_ = edgy(npimg, segment_, debug=False)
    vec(img=npimg, flat_R=segment_, debug=False)
    # reconstruct(npimg.shape, flat_segments=segment_, flat_features=features, file=file, ext=ext, debug=False)


def segmentation2png(segmentation: object, name: object) -> object:
    img = np.zeros((*segmentation.shape, 3), dtype='uint8')
    reg_count = np.max(segmentation)+1

    def get_color(idx):
        D = 256
        o = []
        last = idx
        for i in range(1, 4):
            o.append(last%(D**i))
            last = int(idx/(D**i))
        return np.array(o)
    print("Region count:", reg_count)
    for idx in range(int(reg_count)):
        img[segmentation == idx] = get_color(idx)

    path = os.path.join(cg.INPUT_DIR, name)
    cv2.imwrite(path, img)
    return img


def render_region(fill_segments, lin_grad_fill, npimg):
    mask = np.zeros(fill_segments.shape)
    for reg_idx in range(np.max(fill_segments)+1):
        if reg_idx in lin_grad_fill:
            mask[fill_segments == reg_idx] = 1
    D = np.zeros((*fill_segments.shape, 3))
    W = npimg.shape[1]
    for pid, p in enumerate(mask):
        if p > 0:
            stops, colors = lin_grad_fill[fill_segments[pid]]
            xy = np.array([pid%W, int(p/W)])
            d = stops[-1] - stops[0]
            m = np.linalg.norm(d)
            d = d / m
            proj = d.dot(xy)
            if proj <= 0 : D[pid] = colors[0]
            if proj >= m : D[pid] = colors[-1]
            else:
                for i in range(len(stops)-1):
                    if np.linalg.norm(stops[0] - stops[i]) <= proj < np.linalg.norm(stops[0] - stops[i+1]):
                        r = (proj - np.linalg.norm(stops[0] - stops[i])) / np.linalg.norm(stops[i+1] - stops[i])
                        D[pid] = colors[i] * (1-r) + colors[i+1] * r
                        break

    plt.imshow(D.reshape(*npimg.shape[:2], 3))
    plt.show()


def encode_fill_data(lin_grad_fill, solid_fill, stroke_id):
    msss = []
    for reg_idx in lin_grad_fill:
        if reg_idx != stroke_id:
            stops, colors = lin_grad_fill[reg_idx]
            colors = np.round(255 * colors)
            ds = [np.linalg.norm(stops[0] - s) for s in stops]
            ds = np.cumsum(ds)
            ds = ds / ds[-1]
            mss = [reg_idx, stops[0][0], stops[0][1], stops[-1][0], stops[-1][1]]
            for i, d in enumerate(ds):
                mss.extend([d, colors[i][0], colors[i][1], colors[i][2]])
            mss = ' '.join(['L'] + [str(m) for m in mss])
            msss.append(mss)

    for reg_idx in solid_fill:
        if reg_idx != stroke_id:
            colors = solid_fill[reg_idx]
            colors = np.round(255 * colors)
            mss = ' '.join(['S', str(reg_idx)] + [str(c) for c in colors])
            msss.append(mss)
    for m in msss: print(m)
    return msss


def build4Paper2Pixel(npImg, feature_key, app_config, debug):
    from collections import Counter
    npimg = process_file(npImg)
    segment, features = segments_features(npimg, feature_key)
    segment_ = lin_merge(img=npimg, flat_segment=segment, debug=False)
    if app_config['EDGIFY']: segment_ = edgy(npimg, segment_, debug=False)

    stroke_id = -1
    fill_segments = np.ones(segment_.shape, dtype='int') * stroke_id
    stroke_segment = np.zeros(segment_.shape, dtype='int')

    size_map = Counter(segment_)
    small_region = app_config['SMALL_REGION']
    new_region_idx = 1
    for idx in size_map:
        if size_map[idx] >= small_region or True:
            fill_segments[segment_ == idx] = new_region_idx
            new_region_idx = new_region_idx + 1
        else:
            stroke_segment[segment_ == idx] = 1

    if debug:
        plt.imshow(fill_segments.reshape(npimg.shape[:2]))
        plt.title("Segmentation. {} Regions".format(np.max(fill_segments)+1))
        plt.show()

    lin_grad_fill, solid_fill = vec(img=npimg, flat_R=fill_segments, debug=False)
    for reg_idx in lin_grad_fill:
        print("{} -> {}".format(reg_idx, lin_grad_fill[reg_idx]))
    for reg_idx in solid_fill:
        print("{} -> {}".format(reg_idx, solid_fill[reg_idx]))

    fill_segments[fill_segments==stroke_id] = np.max(fill_segments)+1
    reg_map = segmentation2png(fill_segments.reshape(npimg.shape[:2]), name='region.png')
    assert reg_map.shape == npimg.shape
    stroke_map = (255*stroke_segment).reshape(npimg.shape[:2]).astype(np.uint8)
    path = os.path.join(cg.INPUT_DIR, 'stroke.png')
    cv2.imwrite(path, stroke_map)
    path = os.path.join(cg.INPUT_DIR, 'input_image.png')
    input_image = (npimg*255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
    if debug:
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(input_image)
        axs[1].imshow(reg_map)
        axs[2].imshow(stroke_map)
        plt.show()
    msss = encode_fill_data(lin_grad_fill, solid_fill, stroke_id)
    path = os.path.join(cg.INPUT_DIR, 'reg_fill_data.txt')
    with open(path, 'w') as file:
        msss = [mss+'\n' for mss in msss]
        file.writelines(msss)
    return segment.reshape(npimg.shape[:2]), reg_map, stroke_map


def main(npImg, app_config, debug):
    feature_key=FeatureKey(5.,10./255,5./255)
    msSeg, mergeSeg, strokeSeg = build4Paper2Pixel(npImg, feature_key, app_config, debug=False)
    if debug:
        fig, axs = plt.subplots(1,2, tight_layout=True)
        axs[0].imshow(msSeg)
        axs[1].imshow(mergeSeg)
        fig.suptitle(f'Reg:{len(Counter(mergeSeg.flatten()))}')
        plt.show()
    return msSeg, mergeSeg
    # build(file, feature_key, debug=True)

if __name__ == "__main__":
    app_config = {
        'EDGIFY': True,
        'SMALL_REGION':0
    }
    indir = os.path.join('..', 'vec_assets', 'structured', 'intermediate')
    name = 'hummie.png'
    file = os.path.join(indir, '{}'.format(name))
    img=np.array(Image.open(file))
    main(npImg = img, app_config=app_config, debug=True)