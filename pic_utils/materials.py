from dataclasses import dataclass
import mendeleev


@dataclass(frozen=True)
class Element:
    """
    Represents a chemical element
    """
    name: str
    symbol: str
    atomic_number: int
    relative_atomic_mass: float

    def __repr__(self) -> str:
        return f'{self.name} ({self.symbol}), Z = {self.atomic_number}, Ar = {self.relative_atomic_mass}'

    def __hash__(self) -> int:
        return hash(repr(self))


known_elements = {}


def search_element(name):
    """
    Uses the mendeleev package to search for an element
    """
    el = mendeleev.element(name)
    if el.symbol not in known_elements:
        known_elements[el.symbol] = Element(el.name.lower(), el.symbol, el.atomic_number, el.atomic_weight)
    return known_elements[el.symbol]


class Molecule:
    elements: dict

    def __init__(self, elements: dict) -> None:
        """
        Creates a molecule based on the dictionary of elements, e.g. {'H' : 2, 'O' : 1} for water
        """
        self.elements = {search_element(k): v for k, v in elements.items()}
        self.electrons = sum([element.atomic_number * count for element, count in self.elements.items()])

    def __repr__(self):
        string_array = []
        for element, count in self.elements.items():
            if count == 1:
                string_array.append(element.symbol)
            else:
                string_array.append(f'{element.symbol}{count}')

        return ''.join(string_array)


molecules = {
    'hydrogen': Molecule({'H': 2}),
    'helium': Molecule({'He': 1}),
    'nitrogen': Molecule({'N': 2}),
    'oxygen': Molecule({'O': 2}),
    'methane': Molecule({'H': 4, 'C': 1}),
}


def normalize_shares(share_dict: dict, inplace=False):
    total_weight = sum(share_dict.values())

    if inplace:
        for k in share_dict:
            share_dict[k] /= total_weight
        return share_dict
    else:
        new_dict = {}
        for k in share_dict:
            new_dict[k] = share_dict[k] / total_weight
        return new_dict


def calculate_element_shares(molecule_shares, normalize=False):
    element_shares = {}
    for mol, w in molecule_shares.items():
        for el, count in molecules[mol].elements.items():
            el_share = w * count
            if el in element_shares:
                element_shares[el] += el_share
            else:
                element_shares[el] = el_share
    if normalize:
        normalize_shares(element_shares, inplace=True)

    return element_shares


class Composition:
    def __init__(self, molecule_shares: dict, number_density=None, ionization_levels: dict = None):
        """Creates material composing of several different molecules

        Parameters
        ----------
        molecule_shares : dict
            molecule-float pairs where the float corresponds to the relative share of the molecule.
            Does not have to be normalized.
        number_density : float or similar, optional
            The number density (molecules per unit volume) of the material.
            By default 1, corresponding to calculations in units relative to some reference number density.
        ionization_levels : dict, optional
            element-int pairs for ionization levels for individual elements
            By default, assumes ionization level of 0.
        """
        if number_density is None:
            self.number_density = 1  # calculations per unit of number density
        else:
            self.number_density = number_density

        if ionization_levels is None:
            ionization_levels = {}

        self.molecule_shares = normalize_shares(molecule_shares)
        self.element_shares = calculate_element_shares(self.molecule_shares)

        # store symbols of elements for easier access
        self.element_map = {el.symbol: el for el in self.element_shares}

        self.ionization_levels = {}
        for el in self.element_shares:
            if el.symbol in ionization_levels:
                level = ionization_levels[el.symbol]
                if level > el.atomic_number:
                    raise ValueError(
                        f"Ionization level for {el.symbol} {level} higher than the atomic number {el.atomic_number}"
                    )
                self.ionization_levels[el] = level
            else:
                self.ionization_levels[el] = 0

        self.ionized_electron_share = 0
        self.total_electron_share = 0

        for el in self.element_shares:
            self.total_electron_share += el.atomic_number * self.element_shares[el]
            self.ionized_electron_share += self.ionization_levels[el] * self.element_shares[el]

    def __repr__(self):
        return ', '.join(f'{str(mol)} ({share*100:.3g}%)' for mol, share in self.molecule_shares.items())

    def __iter__(self):
        return iter(self.element_shares)

    def __len__(self):
        return len(self.element_shares)

    def get_number_density(self, element):
        """Calculate number density of an element

        Parameters
        ----------
        element : str or materials.Element
            An element to get the number density in composition for.
            If string is given, "e" and "electron" stand for electrons.

        Returns
        -------
        float or same as number_density
            The number density
        """
        if element == 'e' or element == 'electron':
            return self.number_density * self.ionized_electron_share
        elif isinstance(element, str):
            return self.get_number_density(self.element_map[element])
        else:
            return self.number_density * self.element_shares[element]

    def get_full_ionization_density(self):
        """Calculates the number density of electrons assuming full ionization

        Returns
        -------
        float or same as number_density
            The electron number density
        """
        return self.number_density * self.total_electron_share
