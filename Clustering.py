## author: Anna Sekuła anna.sekula@ikifp.edu.pl


import logging, pandas, rdkit.Chem.Draw, math, rdkit.Chem.AllChem, \
    rdkit.ML.Cluster.Butina, rdkit.ML.Cluster.Murtagh, os, \
    rdkit.ML.Cluster.ClusterVis, PIL.Image


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%I:%M:%S',
                    level=logging.INFO)


def molecules(molecules_file):
    """
    Reading molecules as SMILES from a csv file and converting them to a
        Mol object.
    :param str: Path to a csv file.
    :return: Molecules.
    :rtype: list[rdkit.Chem.rdchem.Mol]
    """
    logging.info('reading molecules...')
    _molecules = []
    for smiles in pandas.read_csv(molecules_file).Smiles:
        molecule = rdkit.Chem.MolFromSmiles(smiles)
        if molecule:
            _molecules.append(molecule)
    return _molecules


def visualiseMolecules(_molecules, filename):
    """
    Saving molecules as a PIL Image.
    Image size is calculated based on molecules count.
    :param list[rdkit.Chem.rdchem.Mol] _molecules: Molecules.
    :param str filename: Name of a file to save molecules in.
    """
    logging.info('visualising molecules...')
    rdkit.Chem.Draw.MolsToGridImage(_molecules,
                                    math.ceil((math.sqrt(len(_molecules)))),
                                    (400, 400)
                                    ).save(filename)


def fingerprints(_molecules):
    """
    Calculating Morgan fingerprints of radius 2 for a set of molecules.
    Fingerprints as bit vectors encode presence or absence of certain
    features in a molecule.
    :param list[rdkit.Chem.rdchem.Mol] _molecules: Molecules.
    :return: Fingerprints as a bit vector.
    :rtype: list[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
    """
    logging.info('calculating fingerprints...')
    return [rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(
                molecule, radius=3)  ## typically 0-3
            for molecule in _molecules]


def distances(_fingerprints, _fingerprints_count):
    """
    Calculating distances between molecules based on Tanimoto similarity.
    Distances matrix is required for clustering.
    Tanimoto similarity is the most popular similarity measure for
    chemical structures. TS = A-and-B / (A + B - A-and-B)
    :param list[<rdkit.DataStructs.cDataStructs.ExplicitBitVect]
        _fingerprints: Fingerprints as a bit vector.
    :param int _fingerprints_count: Number of fingerprints.
    :return: Distances between molecules.
    :rtype: list[float]
    """
    logging.info('calculating distances...')
    distances = []
    for n in range(1, _fingerprints_count):
        distances.extend([1 - similarity
                          for similarity in
                          rdkit.DataStructs.BulkTanimotoSimilarity(
                              _fingerprints[n],
                              _fingerprints[:n])])
    return distances


def cluster(_distances, points, cutoff=0.2):
    """
    Butina clustering is most efficient for large sets of molecules.
    :param list[float]: _distances: Distances between molecules.
    :param int points: Number of points to be used in clustering.
    :param float cutoff: Elements within this range of each other are
        considered to be neighbors.
    :returns: Tuple of clusters, where each cluster is a tuple of ids.
        Cluster centroid is listed first.
    :rtype: tuple[int]
    """
    logging.info('clustering...')
    return rdkit.ML.Cluster.Butina.ClusterData(_distances,
                                               points,
                                               cutoff,
                                               True)


def visualiseClusteredMolecules(_clusters, _molecules):
    """
    Visualising clusters of size greater than 2.
    Saving molecules as a PIL Image.
    Image size is calculated based on molecules count.
    :param tuple[int] clusters: Tuple of clusters, where each cluster is a
        tuple of ids.
    :param list[rdkit.Chem.rdchem.Mol] _molecules: Set of molecules that
        have been clustered.
    """
    logging.info('visualising clustered molecules...')
    for n, cluster in enumerate(_clusters):
        if len(cluster) > 2:
            visualiseMolecules([_molecules[n] for n in cluster],
                               'cluster{}.png'.format(n + 1))


def run(directory):
    """
    All csv files in directory will be processed.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            print()
            logging.info('WORKING ON FILE {}...'.format(filename))
            mols = molecules(filename)
            base_name = os.path.splitext(filename)[0]
            try:
                visualiseMolecules(mols, base_name + '.png')
            except PIL.Image.DecompressionBombError:
                visualiseMolecules(mols[:1000], base_name + '.png')
                logging.warning('image for {} cropped - too many molecules'
                                .format(filename))
            fps = fingerprints(mols)
            fingerprints_count = len(fps)
            dists = distances(fps, fingerprints_count)
            clusters = cluster(dists, fingerprints_count)
            clustered_dir = base_name + '-clustered'
            if not os.path.isdir(clustered_dir):
                os.makedirs(clustered_dir)
            os.chdir(clustered_dir)
            visualiseClusteredMolecules(clusters, mols)
            os.chdir('..')


run('.')
