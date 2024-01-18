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

        if (self.only_selected) and "Selection" not in data.particles.keys():
            raise KeyError("No selection defined")

    @staticmethod
    def get_type_name(data, id):
        # Get the name of a particle type by its ID
        ptype = data.particles["Particle Type"].type_by_id(id)
        name = ptype.name
        if name:
            return name
        return f"Type {id}"

    def get_concentration(self, all_particles_types, selected_indices, selected_neigh_idx):
        # If we are using only selected particle, get concentration from selected atoms + their neighbors
        if self.only_selected:
            unique_neighs = np.unique(selected_neigh_idx)
            combined_mask = np.isin(
                np.arange(len(all_particles_types)),
                np.concatenate([selected_indices, unique_neighs]),
            )

            particle_types = all_particles_types[combined_mask]
        else:
            # if no selection, we use all particles
            particle_types = all_particles_types

        # Calculate the concentration of unique particle types
        unique_types, counts = np.unique(particle_types, return_counts=True)
        return unique_types, counts / len(particle_types)

    @staticmethod
    def get_central_atom_type_mask(unique_types, particles_types):
        # Create a mask for central atom types
        central_atom_type_mask = []
        for atom_type in unique_types:
            central_atom_type_mask.append(np.where(particles_types == atom_type))
        return central_atom_type_mask

    @staticmethod
    def get_wc_from_neigh_in_shell_types(
        neigh_in_shell_types, central_atom_type_mask, c, unique_types
    ):
        # Calculate Warren-Cowley parameters for atoms in shells
        ntypes = len(c)
        neight_in_shell = neigh_in_shell_types.shape[1]
        wc = np.zeros((ntypes, ntypes))

        for i in range(ntypes):
            neight_type_aroud_itype = neigh_in_shell_types[central_atom_type_mask[i]]
            neight_type_aroud_itype_flat = neight_type_aroud_itype.flatten()

            counts = np.bincount(neight_type_aroud_itype_flat, minlength=ntypes + 1)

            pij = counts[unique_types] / (neight_type_aroud_itype.shape[0] * neight_in_shell)

            wc[i, :] = 1 - pij / c

        return wc

    @staticmethod
    def get_particle_types(data):
        particle_types = np.array(data.particles.particle_type)

        return particle_types

    @staticmethod
    def get_nearest_neighbors(data, max_number_of_neigh):
        # Find nearest neighbors for each particle
        finder = NearestNeighborFinder(max_number_of_neigh, data)
        neigh_idx, _ = finder.find_all()

        return neigh_idx

    def create_visualization_tables(self, unique_types, nshells, wc_for_shells, data):
        labels = []
        warrenCowley = []
        idx = list(range(len(unique_types)))

        # Get labels and values for Warren-Cowley parameters
        for m in range(nshells):
            labels.append([])
            warrenCowley.append([])
            for i, j in itertools.combinations_with_replacement(idx, 2):
                namei = self.get_type_name(data, unique_types[i])
                namej = self.get_type_name(data, unique_types[j])
                labels[-1].append(f"{namei}-{namej}")
                warrenCowley[-1].append(wc_for_shells[m, i, j])

        # Create DataTable objects for visualization
        for m in range(nshells):
            table = DataTable(
                title=f"Warren-Cowley parameter (shell={m+1})",
                plot_mode=DataTable.PlotMode.BarChart,
            )
            table.x = table.create_property("i-j pair", data=range(len(labels[m])))
            table.x.types = [ElementType(id=idx, name=l) for idx, l in enumerate(labels[m])]
            table.y = table.create_property(
                f"Warren-Cowley parameter (shell={m+1})", data=warrenCowley[m]
            )
            data.objects.append(table)

    @staticmethod
    def check_symmetry(arr):
        try:
            assert np.allclose(arr.T, arr)
        except:
            print("WARNING: WCs are not exactly symmetric.")

    def modify(self, data: DataCollection, frame: int, **kwargs):
        self.validateInput(data)

        all_particles_types = self.get_particle_types(data)
        ntypes = len(np.unique(all_particles_types))
        nparticles = data.particles.count

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
            neigh_idx_in_shell = selected_neigh_idx[:, self.nneigh[m] : self.nneigh[m + 1]]
            neigh_in_shell_types = all_particles_types[neigh_idx_in_shell]

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
