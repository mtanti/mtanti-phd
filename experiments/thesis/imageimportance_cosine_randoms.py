import numpy as np
import scipy.spatial.distance

np.random.seed(0)

n = 100000

rnn_size = 227
img_size = 268

rnn_vecs = np.random.normal(size=[n, rnn_size])
tgt_img_vecs = np.random.normal(size=[n, img_size])
fol_img_vecs = np.random.normal(size=[n, img_size])

rnd_vecs = np.random.normal(size=[n, rnn_size+img_size])

dists_tgt2fol = [
    scipy.spatial.distance.cosine(
        np.concatenate([rnn_vecs[i], tgt_img_vecs[i]]),
        np.concatenate([rnn_vecs[i], fol_img_vecs[i]])
    )
    for i in range(n)
]

dists_tgt2rnd = [
    scipy.spatial.distance.cosine(
        np.concatenate([rnn_vecs[i], tgt_img_vecs[i]]),
        rnd_vecs[i]
    )
    for i in range(n)
]

print('dists_tgt2fol:', np.mean(dists_tgt2fol))
print('dists_tgt2rnd:', np.mean(dists_tgt2rnd))
