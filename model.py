"""
Classes and methods for a BEM potential problem.
"""

import json
import numpy as np
import plotly
import plotly.graph_objs as go


class Model(object):
    """
    Model for a BEM potential problem.

    A class that stores all the information about the model:
      - Nodes
      - Elements
      - Gauss's points
      - Prescriptions (potentials and flows)
      - Internal points
    """

    name = ""

    class Node:
        """
        Domain nodes.

        A class that stores all properties related to nodes, as coordinates,
        flows and potentials.

        """

        def __init__(self, coords):
            """
            Parameters
            ----------
            coords : list[float]
                 Node's coordinates
            """
            self.coords = np.array(coords)
            self.potentials = None
            """Node's prescribed potentials"""
            self.flows = None
            """Node's prescribed flows"""

        def set_flow(self, flow):
            """Define the node's flow."""
            self.flows = flow

        def set_potential(self, potential):
            """Define the node's potential."""
            self.potentials = potential

    class Element:
        """A class that stores element's connectivity and geometric properties.
        """

        def __init__(self, nodes, dps):
            """
            Parameters
            ----------
            nodes : list[Node]
                Element's initial and final nodes.
            dps : float
                Distance of the singular points.
            """
            self.nodes = nodes
            self.length = (
                (self.nodes[1].coords[0] - self.nodes[0].coords[0]) ** 2 +
                (self.nodes[1].coords[1] - self.nodes[0].coords[1]) ** 2
            ) ** (0.5)
            self.dps = self.length * dps
            """Element's length."""
            self.cos_dir = np.array(
                (self.nodes[1].coords - self.nodes[0].coords) / self.length
            )
            """Element's directions cosines."""
            self.eta = np.array([self.cos_dir[1], -self.cos_dir[0]])
            """Element's normal directions cosines."""
            self.singular_points = (
                nodes[1].coords + nodes[0].coords
            ) / 2 + self.eta * self.dps
            """Element's centroid coordenates."""
            self.centroid = (self.nodes[0].coords + self.nodes[1].coords) / 2
            """Lenght of element's projections over the axis."""
            self.projections = self.length * self.cos_dir / 2

    class InternalPoint:
        """A class for representing internal points."""

        def __init__(self, coords):
            """Constructor."""
            self.coords = np.array(coords)
            self.potentials = [None, None]
            self.flows = [None, None]

        def __str__(self):
            """Representation of internal point as a string"""
            return "(%.4g, %.4g)   %.2e   %.2e   %.2e" % (
                self.coords[0],
                self.coords[1],
                self.potentials,
                self.flows[0],
                self.flows[1]
            )

    def __init__(self, nodes, elements, internal_points, ngauss):
        """
        Parameters
        ----------
        nodes: list[Node]
            Nodes of the model
        elements: list[Element]
            Elements of the model
        internal_points: list[InternalPoint]
            Internal points of the model
        ngauss: int
            Number of Gauss' points of the model
        """
        self.nodes = nodes
        self.elements = elements
        self.internal_points = internal_points
        self.ngauss = ngauss
        # noinspection PyTupleAssignmentBalance
        self.omega, self.ksi = self.gauss(ngauss)

    def show(self):
        """Shows a representation of the model in an interactive plot."""
        # Elementos de contorno
        x = [self.elements[0].nodes[0].coords[0]]
        y = [self.elements[0].nodes[0].coords[1]]
        for element in self.elements:
            x.append(element.nodes[1].coords[0])
            y.append(element.nodes[1].coords[1])
        elements = go.Scattergl(x=x, y=y, name="Elementos de contorno")
        # Nós
        nodes = go.Scatter(
            x=[node.coords[0] for node in self.nodes],
            y=[node.coords[1] for node in self.nodes],
            name="Nós",
            mode="markers",
            marker=dict(size=10),
            text=[
                "Nó: %d<br>Potencial: %.3g<br>Fluxo: %.3g"
                % (i, self.nodes[i].potentials, self.nodes[i].flows)
                for i in range(len(self.nodes))
            ],
            hoverinfo="text",
        )
        # Pontos internos
        internal_points = go.Scatter(
            x=[pi.coords[0] for pi in self.internal_points],
            y=[pi.coords[1] for pi in self.internal_points],
            name="Pontos internos",
            mode="markers",
            text=[
                "Potencial: %.3g<br>Fluxo (X): %.3g<br>Fluxo (Y): %.3g"
                % (
                    self.internal_points[i].potentials,
                    self.internal_points[i].flows[0],
                    self.internal_points[i].flows[1],
                )
                for i in range(len(self.internal_points))
            ],
            hoverinfo="text",
        )
        # Pontos singulares
        singular_points = go.Scatter(
            x=[elem.singular_points[0] for elem in self.elements],
            y=[elem.singular_points[1] for elem in self.elements],
            name="Pontos singulares",
            mode="markers",
        )
        fig = go.Figure(
            data=[elements, nodes, internal_points, singular_points],
            layout=go.Layout(
                title="Model representation",
                xaxis=dict(title="x"),
                yaxis=dict(title="y", scaleanchor="x", scaleratio=1.0),
            ),
        )
        plotly.offline.plot(fig, filename="Model.html")

    def gauss(cls, ngauss):
        """
        Returns the weights (Omegas) and parametric coordinates (ksi) for
        numerical Gauss' integration.

        Parameters
        ----------
        ngauss : int
            Number of desired Gauss' points

        Returns
        -------
        list[float]:
            Weight for Gauss' integration.
        list[float]:
            Parametric coordinates for Gauss' integration.

        Example
        -------
        >>> Model.gauss(1)
        (array([2]), array([0]))

        >>> Model.gauss(2)
        (array([1, 1]), array([ 0.57735027, -0.57735027]))
        """
        if ngauss == 1:
            omega = np.array([2])
            ksi = np.array([0])
        elif ngauss == 2:
            omega = np.array([1, 1])
            ksi = np.array([0.5773502691, -0.5773502691])
        elif ngauss == 3:
            omega = np.array([0.8888888888,
                              0.5555555555,
                              0.5555555555])
            ksi = np.array([0.0,
                            0.7745966692,
                            -0.7745966692])
        elif ngauss == 4:
            omega = np.array([0.6521451548,
                              0.6521451548,
                              0.3478548451,
                              0.3478548451])
            ksi = np.array([0.3399810435,
                            -0.3399810435,
                            0.8611363115,
                            -0.8611363115])
        elif ngauss == 5:
            omega = np.array([0.5688888888,
                              0.4786286704,
                              0.4786286704,
                              0.2369268850,
                              0.2369268850])
            ksi = np.array([0.0,
                            0.5384693101,
                            -0.5384693101,
                            0.9061798459,
                            -0.9061798459])
        return omega, ksi

    def integrate(self, i, j, is_internal):
        # pylint: disable=W0631,R0914,W605
        """
        Computes the influence of a element over a domain/internal point.

        Parameters
        ----------
        i : int
            element's ID
        j : int
            element's ID
        is_internal : boolean
            whether integration is for a domain or internal point

        Returns
        -------
        float
            influence of element j over point i (:math:`H_{i,j}`)
        float
            influence of element j over point i (:math:`G_{i,j}`)
        """
        H, G, Di, Si = 0.0, 0.0, [0.0, 0.0], [0.0, 0.0]
        # Singular element
        if i == j and self.elements[i].dps == 0 and is_internal:
            H = 0.5
            G = (self.elements[j].length / (2 * np.pi)) * (
                (np.log(1 / (self.elements[j].length / 2))) + 1
            )
        else:  # For normal element, Gauss' integration
            for igauss in range(self.ngauss):
                element = self.elements[j]
                gauss_point = (
                    element.centroid + self.ksi[igauss] * element.projections
                )
                if is_internal:
                    d = gauss_point - self.internal_points[i].coords
                else:
                    d = gauss_point - self.elements[i].singular_points
                r = d / np.linalg.norm(d)
                norm_r = np.linalg.norm(d)
                drdn = np.sum(r @ element.eta)
                G = G - (
                    (element.length / (4 * np.pi)) * np.log(norm_r) *
                    self.omega[igauss]
                )
                H = H - (
                    (element.length / (4 * np.pi * norm_r)) *
                    drdn * self.omega[igauss]
                )
                if is_internal is True:
                    Di = (
                        Di +
                        (element.length / (4 * np.pi * norm_r)) *
                        norm_r * self.omega[igauss]
                    )
                    Si = (
                        Si -
                        (element.length / (4 * np.pi * norm_r ** 2)) *
                        ((2 * r * drdn) - element.eta) * self.omega[igauss]
                    )
        if is_internal:
            return H, G, Di, Si
        return H, G

    def solve_boundary(self):
        """Creates the matrices H and G for the model."""
        H = np.zeros((len(self.elements), len(self.elements)))
        G = np.zeros((len(self.elements), len(self.elements)))
        for i in range(len(self.elements)):
            for j in range(len(self.elements)):
                H[i, j], G[i, j] = self.integrate(i, j, False)
        # Verification for the summation of H matrix's lines
        # soma = np.zeros(len(self.elements))
        # for i in range(len(self.elements)):
        #    soma[i] = sum([H[i, j] for j in range(len(self.elements))])
        # print(soma)
        # Swapping matrix's columns
        for j in range(len(self.nodes)):
            if self.nodes[j].flows is None:
                for i in range(len(self.nodes)):
                    H[i, j], G[i, j] = -G[i, j], -H[i, j]
        # Vetor Q que recebe os valores prescritos
        Q = np.zeros(len(self.elements))
        for i in range(len(self.nodes)):
            if self.nodes[i].potentials is None:
                Q[i] = self.nodes[i].flows
            else:
                Q[i] = self.nodes[i].potentials
        # Vetor T de valores independentes do novo sistema HX = T
        T = G @ Q
        # Resolução do sistema algébrico
        X = np.linalg.inv(H) @ T
        # Definicao de erro (Problema Brebbia)
        # solref = [
        #     252.25,
        #     150.02,
        #     47.75,
        #     -52.962,
        #     -48.771,
        #     -52.962,
        #     47.75,
        #     150.02,
        #     252.25,
        #     52.969,
        #     48.737,
        #     52.969,
        # ]
        # verro = X - solref
        # erro = np.linalg.norm(verro)
        # print(erro)
        for i in range(len(self.nodes)):
            if self.nodes[i].potentials is None:
                self.nodes[i].potentials = X[i]
            else:
                self.nodes[i].flows = X[i]

    def solve_domain(self):
        """Computes flow and potentials for the internal points."""
        # pylint: disable=W0631
        Hi = np.zeros((len(self.internal_points), len(self.elements)))
        Gi = np.zeros((len(self.internal_points), len(self.elements)))
        Si = np.zeros((len(self.internal_points), len(self.elements), 2))
        Di = np.zeros((len(self.internal_points), len(self.elements), 2))
        for i in range(len(self.internal_points)):
            for j in range(len(self.elements)):
                # Calculo dos potenciais nos pontos internos
                # noinspection PyTupleAssignmentBalance
                Hi[i, j], Gi[i, j], Di[i, j], Si[i, j] = self.integrate(
                    i, j, True
                )
        U = np.array([node.potentials for node in self.nodes])
        Q = np.array([node.flows for node in self.nodes])
        # Calculation of potentials at internal points
        Ui = -Hi @ U + Gi @ Q
        # Calculation of flows at internal points
        Qi = -Si.reshape(5, 2, 12) @ U.T + Di.reshape(5, 2, 12) @ Q.T
        for i in range(len(self.internal_points)):
            self.internal_points[i].potentials = Ui[i]
            self.internal_points[i].flows = Qi[i]
        print("Coords      U          Q_x          Q_y")
        for ip in self.internal_points:
            print(ip)

    def load_json(cls, file_name):
        """
        Reads a json file and create a `Model` object that contains all model's
         information.

        Parameters
        ----------
        file_name : str
            Input file.

        Returns
        -------
        Model
            A model created from the input file.
        """
        with open(file_name, "r") as f:
            model = json.load(f)
        nodes = [cls.Node(node) for node in model["NODES"]]
        elements = [
            cls.Element(
                [nodes[node[0] - 1], nodes[node[1] - 1]],
                model["SINGULAR_DISTANCE"]
            )
            for node in model["ELEMENTS"]
        ]
        internal_points = [
            cls.InternalPoint(point) for point in model["INTERNALS_POINTS"]
        ]
        for potential in model["POTENTIALS"]:
            nodes[potential[0] - 1].set_potential(potential[1])
        for flow in model["FLOWS"]:
            nodes[flow[0] - 1].set_flow(flow[1])
        return Model(nodes, elements, internal_points, model["NGAUSS"])


if __name__ == "__main__":
    m = Model.load_json("data.json")
    m.solve_boundary()
    m.solve_domain()
    # m.show()
