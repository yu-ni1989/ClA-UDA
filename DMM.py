#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################
import numpy as np

class DMM:
    def __init__(self, dim_s, dim_t):
        self.dim_s = dim_s
        self.dim_t = dim_t
        self.par_matrix = np.zeros([self.dim_s, self.dim_t]) + 1e-12
        self.y_matrix = np.zeros([self.dim_t, self.dim_s]) + 1e-12

    def empirical_matrix(self, v):
        return np.matmul(v, v.T)

    def mutual_matrix(self, v1, v2):
        return np.matmul(v1, v2)

    def weighted_unb_sum_empirical_matrix(self, vs, ass_ind, weights=None):
        if weights == None:
            weights = np.ones(len(ass_ind))
        shape = vs.shape
        if shape[-1] != self.dim_s:
            print('DMM source-domain dimention error in par matrix!')
        # matrix = np.zeros([shape[-1], shape[-1]]) + 1e-12
        count = 0
        for i in ass_ind:
            self.par_matrix = self.par_matrix + weights[count] * self.empirical_matrix(np.matrix(vs[0, i, :]).T)
            count = count + 1

    def regression_matrix(self):
        size = self.dim_s * self.dim_t
        matrix = np.zeros([size, size]) + 1e-12
        size_em = self.par_matrix.shape[0]
        step = size // size_em
        for i in range(step):
            matrix[i * size_em:(i + 1) * size_em, i * size_em:(i + 1) * size_em] = self.par_matrix

        return np.matrix(matrix)

    def weighted_unb_sum_mutual_matrix(self, v1s, v2s, ass_ind_1, ass_ind_2, weights=None):
        if weights == None:
            weights = np.ones(len(ass_ind_1))
        shape_v1 = v1s.shape
        shape_v2 = v2s.shape
        if shape_v1[-1] != self.dim_s or shape_v2[-1] != self.dim_t:
            print('DMM source-domain dimention error in y matrix!')
        count = 0
        for i, j in zip(ass_ind_1, ass_ind_2):
            self.y_matrix = self.y_matrix + weights[count] * self.mutual_matrix(np.matrix(v2s[0, j, :]).T, np.matrix(v1s[0, i, :]))
            count = count + 1

    def regression_y(self):
        shape_m = self.y_matrix.shape
        y = np.zeros([shape_m[0] * shape_m[1], 1]) + 1e-12
        for i in range(shape_m[0]):
            y[i * shape_m[1]: (i + 1) * shape_m[1]] = self.y_matrix[i, :].T

        return np.matrix(y)

    def get_transferring_matrix(self, image_s_vec, image_t_vec, ass_ind_1, ass_ind_2, weights):
        shape_s = image_s_vec.shape
        shape_t = image_t_vec.shape
        if shape_s[-1] != self.dim_s or shape_t[-1] != self.dim_t:
            print('DMM source-domain dimention error !')
        x_matrix_size = [shape_s[-1], shape_t[-1]]
        self.weighted_unb_sum_empirical_matrix(image_s_vec, ass_ind_1, weights)
        x_matrix = self.regression_matrix()
        self.weighted_unb_sum_mutual_matrix(image_s_vec, image_t_vec, ass_ind_1, ass_ind_2, weights)
        y = self.regression_y()

        t_matrix = np.zeros(x_matrix_size)
        if np.linalg.matrix_rank(x_matrix) != x_matrix.shape[-1]:
            return t_matrix, False
        x = np.matmul(x_matrix.I, y)
        x = np.squeeze(np.array(x))
        for i in range(x_matrix_size[-1]):
            t_matrix[:, i] = x[i * x_matrix_size[0]: (i + 1) * x_matrix_size[0]]

        return t_matrix, True

    def reset(self):
        self.par_matrix = np.zeros([self.dim_s, self.dim_t]) + 1e-12
        self.y_matrix = np.zeros([self.dim_t, self.dim_s]) + 1e-12

    def reset_rate(self, rate=0.001):
        self.par_matrix = self.y_matrix * rate
        self.y_matrix = self.y_matrix * rate




