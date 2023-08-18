
from pyrender.material import MetallicRoughnessMaterial
from ..projector.material_coefficients import material_coefficients

class DRRMaterial(MetallicRoughnessMaterial):
    def __init__(self,
                 name=None,
                 normalTexture=None,
                 occlusionTexture=None,
                 emissiveTexture=None,
                 emissiveFactor=None,
                 alphaMode=None,
                 alphaCutoff=None,
                 doubleSided=False,
                 smooth=True,
                 wireframe=False,
                 baseColorFactor=None,
                 baseColorTexture=None,
                 metallicFactor=1.0,
                 roughnessFactor=1.0,
                 metallicRoughnessTexture=None,
                 density=1.0,
                 additive=True,
                 subtractive=False,
                 drrMatName=None
                 ):
        super(DRRMaterial, self).__init__(
            name=name,
            normalTexture=normalTexture,
            occlusionTexture=occlusionTexture,
            emissiveTexture=emissiveTexture,
            emissiveFactor=emissiveFactor,
            alphaMode=alphaMode,
            alphaCutoff=alphaCutoff,
            doubleSided=doubleSided,
            smooth=smooth,
            wireframe=wireframe,
            baseColorFactor=baseColorFactor,
            baseColorTexture=baseColorTexture,
            metallicFactor=metallicFactor,
            roughnessFactor=roughnessFactor,
            metallicRoughnessTexture=metallicRoughnessTexture)

        self.density = density
        self.additive = additive
        self.subtractive = subtractive
        self.drrMatName = drrMatName

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        if value is None:
            value = 1.0
        self._density = float(value)

    @property
    def additive(self):
        return self._additive
    
    @additive.setter
    def additive(self, value):
        self._additive = bool(value)

    @property
    def subtractive(self):
        return self._subtractive
    
    @subtractive.setter
    def subtractive(self, value):
        self._subtractive = bool(value)

    @property
    def drrMatName(self):
        return self._drrMatName
    
    @drrMatName.setter
    def drrMatName(self, value):
        if value is None:
            if self.name is None:
                raise ValueError("Please specify drrMatName")
            if self.name not in material_coefficients:
                raise ValueError(f"Attempted to use material name {self.name} for drrMatName, but it was not found in material coefficients. Please specify drrMatName")
            self._drrMatName = self.name