import itertools

import numpy as np
from ovito.data import DataCollection, DataTable, ElementType, NearestNeighborFinder
from ovito.pipeline import ModifierInterface
from traits.api import Bool, Int, List


class WarrenCowleyParameters(ModifierInterface):
    """Compute the Warren-Cowley parameters.

    ```python
    from ovito.io import import_file
    import WarrenCowleyParameters as WarrenCowleyParameters

    pipeline = import_file("fcc.dump")
    mod = WarrenCowleyParameters(nneigh=[0, 12, 18])
    pipeline.modifiers.append(mod)
    data = pipeline.compute()

    wc_for_shells = data.attributes["Warren-Cowley parameters"]
    print(f"1NN Warren-Cowley parameters: \n {wc_for_shells[0]}")
    print(f"2NN Warren-Cowley parameters: \n {wc_for_shells[1]}")
    ```
    """

    # List of integers representing the maximum number of atoms in shells
    nneigh = List(Int, value=[0, 12], label="Max atoms in shells", minlen=2)

    # Wheather or not to apply it on only selected particles
    only_selected = Bool(False, label="Only selected")

    def modify(self, data: DataCollection, frame: int, **kwargs):
        # Validate input arguments
        validator = InputValidator(data, self.nneigh, self.only_selected)
        validator.validate()

        # Compute Warren-Cowley parameters for each NN shell
        calculator = WarrenCowleyCalculator(data, self.nneigh, self.only_selected)
        wc_for_shells = calculator.calculate_warren_cowley_parameters()

        # Storing the Warren-Cowley parameters as a data attributes
        data.attributes["Warren-Cowley parameters"] = wc_for_shells

        # Ovito DataTables visualization
        visualizer = WarrenCowleyVisualization(data)
        visualizer.create_visualization_tables(
            unique_types=calculator.unique_types,
            nshells=len(self.nneigh) - 1,
            wc_for_shells=wc_for_shells,
        )


###########################################################


class InputValidator:
    def __init__(self, data, nneigh, only_selected):
        self.data = data
        self.nneigh = nneigh
        self.only_selected = only_selected

    def validate(self):
        self._validate_neigh_order()
        self._validate_selection_existence()

    def _validate_neigh_order(self):
        if not np.all(np.diff(self.nneigh) > 0):
            raise ValueError("'Max atoms in shells' must be strictly increasing.")

    def _validate_selection_existence(self):
        if self.only_selected and "Selection" not in self.data.particles.keys():
            raise KeyError("No selection defined in the data")


class WarrenCowleyCalculator:
    def __init__(self, data, nneigh, only_selected):
        self.data = data
        self.nneigh = nneigh
        self.max_number_of_neigh = max(nneigh)
        self.only_selected = only_selected

    def calculate_warren_cowley_parameters(self):
        # Extract particle types and NN idices
        particle_types = self._extract_particle_types()
        neighbor_indices = self._find_neighbor_indices()

        # If using selected particles, compute the WC based on the selected atom and their NN
        nparticles = self.data.particles.count
        selected_indices = (
            np.where(self.data.particles["Selection"] == 1)[0]
            if self.only_selected
            else range(nparticles)
        )
        selected_neigh_idx = neighbor_indices[selected_indices]

        particle_types_selected = (
            particle_types[np.union1d(selected_indices, selected_neigh_idx)]
            if self.only_selected
            else particle_types
        )

        # Get concentration based on selected particles
        unique_types, concentrations = self._calculate_concentration(
            particle_types_selected
        )
        self.unique_types = unique_types

        # Get central atom mask
        central_atom_mask = self._create_central_atom_type_mask(
            unique_types, particle_types[selected_indices]
        )

        ntypes = len(unique_types)
        nshells = len(self.nneigh) - 1
        wc_for_shells = np.zeros((nshells, ntypes, ntypes))

        # Calculate Warren-Cowley parameters for each shell
        for m in range(nshells):
            # Get N shell NN
            neigh_idx_in_shell = selected_neigh_idx[
                :, self.nneigh[m] : self.nneigh[m + 1]
            ]

            # Get atomic types of the NN
            neigh_in_shell_types = particle_types[neigh_idx_in_shell]

            wc = self._compute_wc_params(
                neigh_in_shell_types, central_atom_mask, concentrations, unique_types
            )
            self.verify_symmetry(wc)
            wc_for_shells[m] = wc

        return wc_for_shells

    def _extract_particle_types(self):
        return np.array(self.data.particles.particle_type)

    def _find_neighbor_indices(self):
        finder = NearestNeighborFinder(self.max_number_of_neigh, self.data)
        neighbor_indices, _ = finder.find_all()
        return neighbor_indices

    def _calculate_concentration(self, particle_types):
        unique_types, counts = np.unique(particle_types, return_counts=True)

        return unique_types, counts / len(particle_types)

    def _create_central_atom_type_mask(self, unique_types, particle_types):
        unique_types_array = unique_types[:, np.newaxis]
        return particle_types == unique_types_array

    def _compute_wc_params(
        self, shell_types, central_atom_mask, concentrations, unique_types
    ):
        wc_params = np.zeros((len(concentrations), len(concentrations)))

        Nb = shell_types.shape[1]  # Number of neighbor in shell

        for i in range(len(concentrations)):
            # Flatten list of the type of all neighbors of atom with type i
            neighbor_types = shell_types[central_atom_mask[i]]
            neighbor_types_flat = neighbor_types.flatten()

            # Number of i-X bonds
            neighbor_counts = np.bincount(
                neighbor_types_flat, minlength=len(concentrations) + 1
            )

            pij = neighbor_counts[unique_types] / (neighbor_types.shape[0] * Nb)

            wc_params[i, :] = 1 - pij / concentrations

        return wc_params

    @staticmethod
    def verify_symmetry(parameters):
        if not np.allclose(parameters, parameters.T):
            print("WARNING: The parameters are not symmetric.")


class WarrenCowleyVisualization:
    def __init__(self, data: DataCollection) -> None:
        self.data = data

    def get_type_name(self, id: int) -> str:
        """Get the name of a particle type by its ID"""

        particle_type = self.data.particles["Particle Type"].type_by_id(id)
        return particle_type.name or f"Type {id}"

    def create_visualization_tables(self, unique_types, nshells, wc_for_shells):
        for m in range(nshells):
            labels, values = self._get_labels_and_values(unique_types, wc_for_shells, m)
            self._create_data_table(m, labels, values)

    def _get_labels_and_values(self, unique_types, wc_for_shells, shell_index):
        labels, values = [], []
        idx = range(len(unique_types))

        for i, j in itertools.combinations_with_replacement(idx, 2):
            namei = self.get_type_name(unique_types[i])
            namej = self.get_type_name(unique_types[j])
            labels.append(f"{namei}-{namej}")
            values.append(wc_for_shells[shell_index, i, j])

        return labels, values

    def _create_data_table(self, shell_index, labels, values):
        table = self._create_table(shell_index, labels, values)
        self.data.objects.append(table)

    def _create_table(self, shell_index, labels, values):
        """Creates and configures a data table."""
        table = DataTable(
            title=f"Warren-Cowley parameter (shell={shell_index + 1})",
            plot_mode=DataTable.PlotMode.BarChart,
        )
        table.x = table.create_property("i-j pair", data=range(len(labels)))
        table.x.types = [
            ElementType(id=idx, name=label) for idx, label in enumerate(labels)
        ]
        table.y = table.create_property(
            f"Warren-Cowley parameter (shell={shell_index + 1})", data=values
        )
        return table
