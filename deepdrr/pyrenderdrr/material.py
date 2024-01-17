from pyrender.material import MetallicRoughnessMaterial
from ..projector.material_coefficients import material_coefficients


class DRRMaterial(MetallicRoughnessMaterial):
    _default_densities = {
        "bone": 1.92,
        "soft tissue": 1.0,
        "tissue_soft": 1.0,
        "blood": 1.06,
        "muscle": 1.06,
        "air": 0.0012,
        "iron": 7.87,
        "lung": 0.26,
        "titanium": 4.51,
        "teflon": 2.2,
        "polyethylene": 0.94,
        "concrete": 2.3,
    }

    def __init__(
        self,
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
        layer=0,
        drrMatName=None,
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
            metallicRoughnessTexture=metallicRoughnessTexture,
        )

        self.density = density
        self.additive = additive
        self.subtractive = subtractive
        self.layer = layer
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
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, value):
        if value is None:
            value = 0
        self._layer = int(value)

    @property
    def drrMatName(self):
        return self._drrMatName

    @drrMatName.setter
    def drrMatName(self, value):
        if value is None:
            if self.name is None:
                raise ValueError("Please specify drrMatName")
            if self.name not in material_coefficients:
                raise ValueError(
                    f"Attempted to use material name {self.name} for drrMatName, but it was not found in material coefficients. Please specify drrMatName"
                )
            self._drrMatName = self.name

    @classmethod
    def from_name(cls, name: str, **kwargs):
        """Make a material with default parameters."""
        if name not in material_coefficients:
            raise ValueError(
                f"Attempted to use material name {name}, but it was not found in material coefficients. Please specify drrMatName"
            )
        density = cls._default_densities[name]
        return cls(name=name, density=density, **kwargs)

    @classmethod
    def Iron(cls, **kwargs):
        return cls.from_name("iron", **kwargs)

    @classmethod
    def Bone(cls, **kwargs):
        return cls.from_name("bone", **kwargs)

    @classmethod
    def SoftTissue(cls, **kwargs):
        return cls.from_name("soft tissue", **kwargs)

    @classmethod
    def TissueSoft(cls, **kwargs):
        return cls.from_name("tissue_soft", **kwargs)

    @classmethod
    def Blood(cls, **kwargs):
        return cls.from_name("blood", **kwargs)

    @classmethod
    def Muscle(cls, **kwargs):
        return cls.from_name("muscle", **kwargs)

    @classmethod
    def Air(cls, **kwargs):
        return cls.from_name("air", **kwargs)

    @classmethod
    def Lung(cls, **kwargs):
        return cls.from_name("lung", **kwargs)

    @classmethod
    def Titanium(cls, **kwargs):
        return cls.from_name("titanium", **kwargs)

    @classmethod
    def Teflon(cls, **kwargs):
        return cls.from_name("teflon", **kwargs)

    @classmethod
    def Polyethylene(cls, **kwargs):
        return cls.from_name("polyethylene", **kwargs)

    @classmethod
    def Concrete(cls, **kwargs):
        return cls.from_name("concrete", **kwargs)
