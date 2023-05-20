import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod




class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()
        self.resized_rgb_add = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
            self.seams_rgb_add = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        self.add_col = np.zeros((self.h,self.w), dtype=int)
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)

        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))
        

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        
        gray = np.matmul(np_img, self.gs_weights)
        gray = np.pad(gray, ((1,1),(1,1),(0,0)), 'constant', constant_values= 0.5)
        
        return gray

    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            In order to calculate a gradient of a pixel, only its neighborhood is required.
        """
        """## get the grayscale image and change its shape to (h,w)
        i = self.resized_gs.squeeze()

        #add padding on one side each time , and calculate one side derivative.
        padded_i = np.pad(i, ((0,0),(1,0)), mode='constant', constant_values=0)
        gx = np.diff(padded_i, axis=1)
        padded_i = np.pad(i, ((1,0),(0,0)), mode='constant', constant_values=0)
        gy = np.diff(padded_i, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Normalize the filtered image to the range [0, 1]
        magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

        return magnitude"""

        ## get the grayscale image and change its shape to (h,w)
        i = self.resized_gs.squeeze()
        #add padding on one side each time , and calculate one side derivative.
        padded_i = np.pad(i, ((0,0),(0,1)), mode='constant', constant_values=0.5)
        gx = np.diff(padded_i, axis=1)
        gx[:,-1] = i[:,-1]-i[:,-2]

        padded_i = np.pad(i, ((0,1),(0,0)), mode='constant', constant_values=0.5)
        gy = np.diff(padded_i, axis=0)
        gy[-1,:] = i[-1,:] - i[-2,:]
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Normalize the filtered image to the range [0, 1]
        magnitude[magnitude>1] = 1

        return magnitude

        
    

    

        
    def calc_M(self):
        pass
             
    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        pass

    def init_mats(self):
        pass

    def update_ref_mat(self):
        pass

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(self.path)

    @staticmethod
    def load_image(img_path):
        return np.asarray(Image.open(img_path)).astype('float32') / 255.0


class ColumnSeamImage(SeamImage):
    """ Column SeamImage.
    This class stores and implements all required data and algorithmics from implementing the "column" version of the seam carving algorithm.
    """
    def __init__(self, *args, **kwargs):
        """ ColumnSeamImage initialization.
        """
        super().__init__(*args, **kwargs)

        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture, but with the additional constraint:
            - A seam must be a column. That is, the set of seams S is simply columns of M. 
            - implement forward-looking cost

        Returns:
            A "column" energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            The formula of calculation M is as taught, but with certain terms omitted.
            You might find the function 'np.roll' useful.
        """
        M = np.zeros((self.E.shape[0]-2, self.E.shape[1]-2), dtype = np.float32 )
        
        padded_i = np.pad(self.resized_gs.squeeze(), ((1,1),(0,0)), mode='constant', constant_values=0)
        cv = np.abs((np.roll(padded_i, -1, axis = 1) - np.roll(padded_i, 1 , axis=1) ))
        cv = cv[:,1:-1]
        cv = cv[:-2,:]

        M[0,:] = self.E[1,1:-1]
        for i in range(1,M.shape[0]):
            M[i,:] = cv[i,:] + M[i-1,:] + self.E[i,1:-1]

        


        return M

    def seams_removal(self, num_remove: int, flag = False):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matric
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) seam backtracking: calculates the actual indices of the seam
            iii) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            iv) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to support:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """

        for i in range(num_remove):
            # find index of the next seam to remove
            col_idx = np.argmin(self.M[-1,2:-2])+2
            
            # change the seam to zeros in mask
            if flag:
                self.cumm_mask[ :,self.idx_map_v[col_idx,1]] = 0
            else:
                self.cumm_mask[:, self.idx_map_h[1,col_idx]] = 0
            #update E and M
            self.update_E(col_idx)
            self.update_M(col_idx)
            #create the new resized image
            self.resized_rgb = np.hstack((self.resized_rgb[:,0:col_idx] , self.resized_rgb[:, col_idx+1:]))
            self.resized_gs = np.hstack((self.resized_gs[:,0:col_idx] , self.resized_gs[:, col_idx+1:]))
            if flag:
                roll = self.idx_map_v[col_idx:, :]
                roll = np.roll(roll, -1, axis = 0)
                self.idx_map_v[col_idx:, :] = roll
            else:
                roll = self.idx_map_h[:, col_idx:]
                roll = np.roll(roll, -1, axis = 1)
                self.idx_map_h[:, col_idx:] = roll
        


        mask = self.cumm_mask[1:-1,1:-1,0] == 0

        self.seams_rgb[:,:,0][mask] = 1
        self.seams_rgb[:,:,1][mask]= 0
        self.seams_rgb[:,:,2][mask] = 0
            



        return None

    def update_E(self, seam_idx):

        i = np.hstack((self.resized_gs.squeeze()[:,seam_idx-1,None],self.resized_gs.squeeze()[:,seam_idx+1,None]))
        padded_i = np.pad(i, ((0,0),(0,1)), mode='constant', constant_values=0)
        gx = np.diff(padded_i, axis=1)
        padded_i = np.pad(i, ((0,1),(0,0)), mode='constant', constant_values=0)
        gy = np.diff(padded_i, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        col = magnitude[:,1]
        col[col>1] =1
        self.E[:,seam_idx-1] = col
        self.E = np.hstack((self.E[:,:seam_idx],self.E[:,seam_idx+1:]))

        
        
        
        return None

    def update_M(self, seam_idx):

        i = np.hstack((self.resized_gs.squeeze()[:,seam_idx-2:seam_idx],self.resized_gs.squeeze()[:,seam_idx+1:seam_idx+3]))
        e = np.hstack((self.E[:,seam_idx-1,None], self.E[:,seam_idx,None]))
        M = np.zeros((e.shape[0]-2, 2), dtype = np.float32 )
        padded_i = np.pad(i , ((0,0),(1,1)), mode='constant', constant_values=0)

        cv = np.abs((np.roll(padded_i, -1, axis = 1) - np.roll(padded_i, 1 , axis=1) ))
        
        cv = cv[:,2:-2]
        cv= np.roll(cv, 1, axis = 0)
        cv[0,:]= 0 

        M[0,:] = e[1,:]
        for i in range(1,M.shape[0]):
            M[i,:] = cv[i,:] + M[i-1,:] + e[i+1, :]


    
        self.M = np.hstack((self.M[:,:seam_idx],self.M[:,seam_idx+1:])) 


        
        if seam_idx == self.M.shape[1] :
            self.M[:,-1] = M[:,0]
        elif seam_idx == 0 :
            self.M[:,0] = M[:,1]
        else:
            self.M[:,seam_idx-1:seam_idx+1] = M 

        return None

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """

        self.resized_rgb = np.rot90(self.resized_rgb,1,(0,1))
        self.seams_rgb= np.rot90(self.seams_rgb,1,(0,1))
        self.resized_gs = np.rot90(self.resized_gs, 1, (0,1))
        self.cumm_mask = np.rot90(self.cumm_mask,1,(0,1))
        
        self.E = np.rot90(self.E,k=1)
        self.M = self.calc_M()

        self.seams_removal(num_remove, True)

        self.resized_rgb = np.rot90(self.resized_rgb,-1,(0,1))
        self.seams_rgb= np.rot90(self.seams_rgb,-1,(0,1))
        self.resized_gs = np.rot90(self.resized_gs, -1, (0,1))
        self.cumm_mask = np.rot90(self.cumm_mask,-1,(0,1))
        self.E = np.rot90(self.E,k=1)
        self.M = self.calc_M()


        return None

    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of vertical seam to be removed
        """

        self.seams_removal(num_remove)

        return None

    def backtrack_seam(self):
        """ Backtracks a seam for Column Seam Carving method
        """
        raise NotImplementedError("TODO: Implement SeamImage.backtrack_seam")

    def remove_seam(self):
        """ Removes a seam for self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        raise NotImplementedError("TODO: Implement SeamImage.remove_seam")


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        self.gs = np.matmul(self.rgb, self.gs_weights).squeeze()
        self.resized_gs = self.gs.copy()
        self.E = self.calc_E()
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)
    
    def calc_E(self):
        ## get the grayscale image and change its shape to (h,w)
        i = self.resized_gs
        #add padding on one side each time , and calculate one side derivative.
        padded_i = np.pad(i, ((0,0),(0,1)), mode='constant', constant_values=0.5)
        gx = np.diff(padded_i, axis=1)
        gx[:,-1] = i[:,-1]-i[:,-2]

        padded_i = np.pad(i, ((0,1),(0,0)), mode='constant', constant_values=0.5)
        gy = np.diff(padded_i, axis=0)
        gy[-1,:] = i[-1,:] - i[-2,:]
        magnitude = np.sqrt(gx**2 + gy**2)
        # Normalize the filtered image to the range [0, 1]
        magnitude[magnitude>1] = 1

        return magnitude


        
        
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        
        M = np.zeros((self.E.shape[0], self.E.shape[1]), dtype = np.float32 )
        
        padded_i = np.pad(self.resized_gs, ((0,1),(0,0)), mode='constant', constant_values=0)
        cv = np.abs((np.roll(padded_i, -1, axis = 1) - np.roll(padded_i, 1 , axis=1) ))
        cv = cv[:-1,:]
        cv[:,[0,-1]] = 0
        
         
        padded_i = np.pad(self.resized_gs, ((1,1),(1,1)), mode='constant', constant_values=0)
        cr = np.abs((np.roll(padded_i, -1, axis = 1) - np.roll(padded_i, 1 , axis=0) ))
        cr = cr[1:-1,1:-1]
        cr[:,-1]=0
        cr = cr + cv

        cl = np.abs((np.roll(padded_i, 1, axis = 0) - np.roll(padded_i, 1 , axis=1) ))
        cl = cl[1:-1,1:-1]
        cl[:,0]=0
        cl = cl+cv

        M[0,:] = self.E[0,:]
        
        for i in range(1,M.shape[0]):
            right_shift = np.roll(M[i-1,:], 1 )
            right_shift[0] = 0
            left_shift = np.roll(M[i-1,:], -1 )
            left_shift[-1] = 0

            M[i,:] =   self.E[i,:] + np.min(np.array([right_shift + cl[i,:], left_shift +cr[i,:], M[i-1,:] + cv[i,:]]), axis = 0)

            M[i,0] =  self.E[i,0] + np.min( np.array([left_shift[0] + cr[i,0], M[i-1,0] + cv[i,0]]))
            M[i,-1] =  self.E[i,-1] + np.min( np.array([right_shift[-1] + cl[i,0], M[i-1,-1] + cv[i,-1]]))

        
        


        return M

        

    def seams_removal(self, num_remove: int,add, ver = True):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        seam = None   
        for i in range(num_remove):
            seam = self.backtrack_seam(ver,add)
            self.E = self.calc_E()
            self.M = self.calc_M()


        return seam
    



            


    def seams_removal_horizontal(self, num_remove,add=False):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
    

        self.resized_rgb = np.transpose(self.resized_rgb, (1,0,2))
        self.seams_rgb= np.transpose(self.seams_rgb, (1,0,2))
        self.resized_gs = self.resized_gs.T
        
        self.E = self.calc_E()
        self.M = self.calc_M()

        temp = self.idx_map_h
        self.idx_map_h = self.idx_map_v.T
        self.idx_map_v = temp.T

        seam = self.seams_removal(num_remove,add, ver = False)

        self.resized_rgb = np.transpose(self.resized_rgb, (1,0,2))
        self.seams_rgb= np.transpose(self.seams_rgb, (1,0,2))
        self.resized_gs = self.resized_gs.T

        temp = self.idx_map_h
        self.idx_map_h = self.idx_map_v.T
        self.idx_map_v = temp.T

        self.E = self.calc_E()
        self.M = self.calc_M()

        return seam

    def seams_removal_vertical(self, num_remove, add=False):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        seam = self.seams_removal(num_remove,add)

        return seam
        

    def backtrack_seam(self, ver,add):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        padded_i = np.pad(self.resized_gs, ((0,1),(0,0)), mode='constant', constant_values=0)
        cv = np.abs((np.roll(padded_i, -1, axis = 1) - np.roll(padded_i, 1 , axis=1) ))
        cv = cv[:-1,:]
        cv[:,[0,-1]] = 0
        
        
        padded_i = np.pad(self.resized_gs, ((1,1),(1,1)), mode='constant', constant_values=0)
        cl = np.abs((np.roll(padded_i, 1, axis = 0) - np.roll(padded_i, 1 , axis=1) ))
        cl = cl[1:-1,1:-1]
        cl[:,0]=0
        cl = cl+cv

        cr = np.abs((np.roll(padded_i, -1, axis = 1) - np.roll(padded_i, 1 , axis=0) ))
        cr = cr[1:-1,1:-1]
        cr[:,-1]=0
        cr = cr + cv


        seam = np.zeros((self.M.shape[0]), dtype=int)
        seam[0] = np.argmin(self.M[-1,:])
        s = 1

        for i in range(self.M.shape[0]-2, -1 ,-1):
            
            
            if seam[s-1] == 0:

                if self.M[i,seam[s-1]] + cv[i+1,seam[s-1]] < self.M[i,seam[s-1]+1] + cr[i+1,seam[s-1]]:
                    seam[s] = seam[s-1]
                else:
                    seam[s] = seam[s-1] + 1

                
            elif seam[s-1] == self.M.shape[1]-1:
                if self.M[i,seam[s-1]] + cv[i+1,seam[s-1]] < self.M[i,seam[s-1]-1] + cl[i+1,seam[s-1]]:
                    seam[s] = seam[s-1]
                else:
                    seam[s] = seam[s-1] - 1

                
            else:
                if self.M[i,seam[s-1]] + cv[i+1,seam[s-1]] < self.M[i,seam[s-1]+1] + cr[i+1,seam[s-1]] and self.M[i,seam[s-1]] + cv[i+1,seam[s-1]] < self.M[i,seam[s-1]-1] + cl[i+1,seam[s-1]]:
                    seam[s] = seam[s-1]
                elif self.M[i,seam[s-1]+1] + cr[i+1,seam[s-1]] < self.M[i,seam[s-1]-1] + cl[i+1,seam[s-1]]:
                    seam[s] = seam[s-1] + 1
                else:
                    seam[s] = seam[s-1] - 1


            #find indicies in original image
            row = self.idx_map_v[i+1,seam[s-1]]
            col = self.idx_map_h[i+1,seam[s-1]]
            #color the seam in red
            if not add:
                #color the seam in red
                self.seams_rgb[row,col,0] = 1
                self.seams_rgb[row,col,1] = 0
                self.seams_rgb[row,col,2] = 0
            else:
                self.seams_rgb_add[row,col,0] = 0
                self.seams_rgb_add[row,col,1] = 1
                self.seams_rgb_add[row,col,2] = 0
            #update indicies matrix
            self.idx_map_v[i+1,:] = np.append(self.idx_map_v[i+1,:seam[s-1]] ,np.roll(self.idx_map_v[i+1,seam[s-1]:],-1))
            self.idx_map_h[i+1,:] = np.append(self.idx_map_h[i+1,:seam[s-1]] ,np.roll(self.idx_map_h[i+1,seam[s-1]:],-1))

            #move pixels to last coloumn in gs image
            self.resized_gs[i+1,:] = np.append(self.resized_gs[i+1,:seam[s-1]], np.roll(self.resized_gs[i+1,seam[s-1]:],-1)) 
            
            #move pixels to last coloumn in rgb image
            self.resized_rgb[i+1,:,0] = np.append(self.resized_rgb[i+1,:seam[s-1],0], np.roll(self.resized_rgb[i+1,seam[s-1]:,0], -1 ))
            self.resized_rgb[i+1,:,1] = np.append(self.resized_rgb[i+1,:seam[s-1],1], np.roll(self.resized_rgb[i+1,seam[s-1]:,1], -1 ))
            self.resized_rgb[i+1,:,2] = np.append(self.resized_rgb[i+1,:seam[s-1],2], np.roll(self.resized_rgb[i+1,seam[s-1]:,2], -1 ))

          
            s+=1
        
        #roll first row : 
        self.resized_gs[0,:] = np.append(self.resized_gs[0,:seam[-1]], np.roll(self.resized_gs[0,seam[-1]:],-1)) 
        self.idx_map_v[0,:] = np.append(self.idx_map_v[0,:seam[s-1]] ,np.roll(self.idx_map_v[0,seam[s-1]:],-1))
        self.idx_map_h[0,:] = np.append(self.idx_map_h[0,:seam[s-1]] ,np.roll(self.idx_map_h[0,seam[s-1]:],-1))
        #delete last coloumn from gs image
        self.resized_gs = self.resized_gs[:,:-1]
        #delete last coloumn from rgb image
        self.resized_rgb = self.resized_rgb[:,:-1,:]

        #update indicies matrix
        self.idx_map_v = self.idx_map_v[:,:-1]
        self.idx_map_h = self.idx_map_h[:,:-1]

        
        return seam

    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        raise NotImplementedError("TODO: Implement SeamImage.remove_seam")
    
    def seams_addition(self, num_add: int, ver = True):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        seams = np.zeros((num_add, self.M.shape[0]), dtype= int)

        if ver:
            seams = np.zeros((num_add, self.M.shape[0]), dtype= int)
            for i in range(num_add):
                seams[i,:] = self.seams_removal_vertical(1,add = True)
        else:
            seams = np.zeros((num_add, self.resized_rgb_add.shape[0]), dtype= int)
            for i in range(num_add):
                seams[i,:] = self.seams_removal_horizontal(1,add = True)

        return seams
    
    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.seams_rgb_add = np.transpose(self.seams_rgb_add, (1,0,2))
        self.resized_rgb_add = np.transpose(self.resized_rgb_add, (1,0,2))
        seams = self.seams_addition(num_add, ver = False)
        self.seams_rgb_add = np.transpose(self.seams_rgb_add, (1,0,2))
        self.resized_rgb_add = np.transpose(self.resized_rgb_add, (1,0,2))

        resized_img = np.zeros((self.resized_rgb_add.shape[0]+num_add, self.resized_rgb_add.shape[1],3))
        resized_img[:self.resized_rgb_add.shape[0],:] = self.resized_rgb_add
        for i in range(num_add):
            for j in range(seams[i,:].shape[0]):
                col = self.resized_rgb_add.shape[1] -j -1
                row = seams[i,j] + self.add_col[i,seams[i,j]]

                #add seams pixel
                resized_img[:,col,0] = np.append(resized_img[:row,col,0], np.roll(resized_img[row:,col,0],1))
                resized_img[row,col,0] = resized_img[row+1,col,0]
                resized_img[:,col,1] = np.append(resized_img[:row,col,1], np.roll(resized_img[row:,col,1],1))
                resized_img[row,col,1] = resized_img[row+1,col,1]
                resized_img[:,col,2] = np.append(resized_img[:row,col,2], np.roll(resized_img[row:,col,2],1))
                resized_img[row,col,2] = resized_img[row+1,col,2]



                self.add_col[seams[i,j]+1:,col] +=1

            self.resized_rgb_add = resized_img





        return None

    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        seams = self.seams_addition(num_add)
        resized_img = np.zeros((self.resized_rgb_add.shape[0], self.resized_rgb_add.shape[1]+num_add,3))
        resized_img[:,:self.resized_rgb_add.shape[1]] = self.resized_rgb_add
        for i in range(num_add):
            for j in range(seams[i,:].shape[0]):
                #find indicies in resized matrix
                row = self.resized_rgb_add.shape[0] -1 -j 
                col = seams[i,j] + self.add_col[row,seams[i,j]]
                #add seams pixel
                resized_img[row,:,0] = np.append(resized_img[row,:col,0], np.roll(resized_img[row,col:,0],1))
                resized_img[row,col,0] = resized_img[row,col+1,0]
                resized_img[row,:,1] = np.append(resized_img[row,:col,1], np.roll(resized_img[row,col:,1],1))
                resized_img[row,col,1] = resized_img[row,col+1,1]
                resized_img[row,:,2] = np.append(resized_img[row,:col,2], np.roll(resized_img[row,col:,2],1))
                resized_img[row,col,2] = resized_img[row,col+1,2]

                #update index matrix
                self.add_col[row,seams[i,j]+1:] +=1
        self.resized_rgb_add = resized_img




        return None

    @staticmethod
    # @jit(nopython=True)
    def calc_bt_mat(M, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.
        
        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")

def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    new_shape = np.array([orig_shape[0]*scale_factors[0] ,orig_shape[1]*scale_factors[1]], dtype = int)
                
    return new_shape

def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """

    seam_img.seams_removal_vertical(shapes[0][1] - shapes[1][1])
    seam_img.seams_removal_horizontal(shapes[0][0] - shapes[1][0])
    
    return seam_img.resized_rgb

def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image






