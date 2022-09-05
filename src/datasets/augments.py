from albumentations import ImageOnlyTransform
import numpy as np
import cv2

AUGS = []

try:
    import staintools
    class StainAug(ImageOnlyTransform):
        def __init__(self, always_apply = False, p = 0.5):
            super(StainAug, self).__init__(always_apply=always_apply, p=p)
            self.augmentor = staintools.StainAugmentor(
                method = 'vahadane', 
                sigma1 = 0.2, 
                sigma2 = 0.2)

        def apply(self, img, **params):
            self.augmentor.fit(img)
            img = self.augmentor.pop()
            return np.clip(img.round(), 0, 255).astype(np.uint8)
    AUGS.append(StainAug)

    from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
    from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
    from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator
    from staintools.miscellaneous.get_concentrations import get_concentrations

    class StainNormAug(ImageOnlyTransform):
        def __init__(self, method = 'vahadane', sigma1 = 0.2, sigma2 = 0.2, aug = 0.5, always_apply = False, p = 0.5):
            super(StainNormAug, self).__init__(always_apply=always_apply, p=p)
            if method.lower() == 'macenko':
                self.extractor = MacenkoStainExtractor
            elif method.lower() == 'vahadane':
                self.extractor = VahadaneStainExtractor
            else:
                raise Exception('Method not recognized.')
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.aug = aug

        @property
        def maxC_target(self):
            return np.array([[2.80235547, 1.64160326]])

        @property
        def matrix_target(self):
            return np.array([[0.51196973, 0.76052588, 0.39456241],
                            [0.18823047, 0.84396271, 0.46774341]])
            
        def apply(self, img, **params):
            maxC_target = self.maxC_target
            stain_matrix_target = self.matrix_target
            I = img
            # standardize
            I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
            L_float = I_LAB[:, :, 0].astype(float)
            p = np.percentile(L_float, 95)
            I_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
            I = cv2.cvtColor(I_LAB, cv2.COLOR_LAB2RGB)
            # norm
            stain_matrix_source = self.extractor.get_stain_matrix(I)
            source_concentrations = get_concentrations(I, stain_matrix_source)
            maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
            source_concentrations *= (maxC_target / maxC_source)
            normed = 255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target))
            normed = normed.reshape(I.shape).astype(np.uint8)
            I = normed
            # aug
            if np.random.random() < self.aug:
                image_shape = I.shape
                stain_matrix = stain_matrix_target
                n_stains = source_concentrations.shape[1]
                tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I).ravel()

                for i in range(n_stains):
                    alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
                    beta = np.random.uniform(-self.sigma2, self.sigma2)
                    source_concentrations[tissue_mask, i] *= alpha
                    source_concentrations[tissue_mask, i] += beta

                I = 255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix))
                I = I.reshape(image_shape)
                I = np.clip(I, 0, 255).astype(np.uint8)
            return I
    AUGS.append(StainNormAug)
except:
    pass
