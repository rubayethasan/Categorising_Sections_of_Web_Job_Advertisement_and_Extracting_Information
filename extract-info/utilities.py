import os
import re
import string
import multiprocessing
import bs4
import pandas as pd
import numpy as np
import scipy

from ahocorasick import Automaton
from currency_converter import CurrencyConverter
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from scipy import spatial
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pycountry


# region --- Tree structure ---


class Tree:
    """
    Tree structure used for the NACE and ISCO trees
    """

    def __init__(self, root_id='R'):
        """
        :param root_id: id of the root if the tree, also starting character for any node id in the tree
        :type root_id: str
        """
        self.root = Node(root_id)

    def add_node(self, node):
        """
        Add a node to the tree. The node gets placed based on its id; any intermediate nodes are created automatically.

        :param node: Node to be added to the tree.
        :type node: Node
        """
        node_id = node.id.split('.')[0]
        parent = self.root
        for i in range(len(node_id)):
            code = node_id[i]
            if code in parent.children:
                parent = parent.children[code]
            else:
                n = Node(code, parent=parent)
                parent = n
        node.id = node.id.split('.')[1]
        node.add_father(parent)

    def propagate(self, attr, node=None):
        """
        Propagates an attribute up a tree. The attribute NEEDS to be a dictionary of the format {name: 1} at the leaves.

        :param attr: name of the attribute to propagate
        :type attr: str
        :param node: used for recursion purposes only, do not use when calling this method otherwise.
        :type node: Node
        :return: returns a dictionary with all the names and counts for each name. i.e. {name1: 100, name2:120, ...}
        :rtype: dict
        """
        if node is None:
            node = self.root
        if not hasattr(node, 'essential_skl'):
            node.update_attributes(essential_skl={}, optional_skl={})
        if len(node.children) != 0:
            for child in node.children.values():
                attrs = self.propagate(attr, child)
                for a in attrs:
                    if a not in node.__getattribute__(attr):
                        node.__getattribute__(attr)[a] = attrs[a]
                    else:
                        node.__getattribute__(attr)[a] += attrs[a]
        return node.__getattribute__(attr)


class Node:
    """
    Node used in the tree structure
    """

    def __init__(self, ID, parent=None, **kwargs):
        """
        :param ID: id of the node, which also determines the position in the tree.
        :type ID: str
        :param parent: parent of the current node in the tree.
        :type parent: Node
        :param kwargs: any attributes to initialise the node with.
        :type kwargs: dict
        """
        self.children = {}
        self.id = ID
        self.parent = None
        self.code = None
        self.__dict__.update(kwargs)
        self.add_father(parent)

    def add_father(self, parent):
        """
        Calculates the code attribute, adds self as child of the parent node.

        :param parent: parent of the current node in the tree.
        :type parent: Node
        """
        self.parent = parent
        n = parent
        self.code = []
        while n is not None:
            self.code.append(n.id[-1])
            n = n.parent
        self.code.reverse()
        self.code = ''.join(self.code)

        if parent is not None:
            self.parent.children[self.id] = self

    def update_attributes(self, **attr):
        """
        Updates/adds attributes

        :param attr: attributes to update, should be in the format {attr1:value1, attr2,:value2 ...}
        :type attr: dict
        """
        self.__dict__.update(attr)

    def copy(self):
        """
        :return: deep copy of the object
        :rtype: Node
        """
        import copy
        return copy.deepcopy(self)


# endregion


# region --- Preprocessing ---


class Preprocessor:
    """
    Class used to preprocess the data
    """

    def __init__(self, mode=2, ordered=True, verbose=True):
        """
        :param mode: 0 - strip HTML only; 1 - minimal cleaning; 2 - delete everything between parenthesis
        :type mode: int
        :param ordered: whether to return values in the same order, False is a bit quicker and uses a bit less memory.
        :type ordered: bool
        :param verbose: whether to print the progress bar to console
        :type verbose: bool
        """
        self._mode = mode
        self._ordered = ordered
        self._verbose = verbose

    def clean(self, text, verbose=None):
        """
        Clean the input text with the settings from mode

        :param text: the text to be cleaned as a list of str
        :type text: list
        :return: cleaned text, in the same order
        :rtype: list
        """
        if verbose is not None:
            self._verbose = verbose
        return self._run(text, self._clean)

    def _run(self, data, func):
        """
        Runs a function in parallel on all available cores

        :param data: the input data, should be a list of input values for the function
        :type data: list
        :param func: the function to run on all the data
        :type func: callable
        :return: the data once the function has been applied to each element of the list
        :rtype: list
        """
        with multiprocessing.Pool(processes=os.cpu_count()) as p:
            if self._verbose:
                with tqdm(total=len(data), unit=' sentences') as p_bar:
                    if self._ordered:
                        for i, res in enumerate(p.imap(func, data)):
                            p_bar.update()
                            data[i] = res
                    else:
                        for i, res in enumerate(p.imap_unordered(func, data)):
                            p_bar.update()
                            data[i] = res
            else:
                if self._ordered:
                    for i, res in enumerate(p.imap(func, data)):
                        data[i] = res
                else:
                    for i, res in enumerate(p.imap_unordered(func, data)):
                        data[i] = res
        return data

    def _clean(self, text):
        """
        Clean text using different modes

        :param text: text to be cleaned
        :type text: str
        :return: cleaned text
        :rtype: str
        """
        if type(text) != str:
            return ''

        soup = bs4.BeautifulSoup(text, features="html5lib")
        for script in soup(["script", "style"]):
            script.decompose()

        strips = list(soup.stripped_strings)

        text = " ".join(strips)
        text = ''.join(x for x in text if x in string.printable)
        text = text.replace('col-wide', '')

        if self._mode > 0:

            text = text.replace('?', '').replace('-', '').replace(',', '')
            text = text.lower()

            if self._mode == 2:
                text = re.sub("[\(\[].*?[\)\]]", "", text)

            text = text.replace('(', '').replace(')', '')

        text = text.replace('  ', ' ')

        return text


# endregion


# region --- ESCO, ISCO, NACE ---


NACE_SEC_TO_CODE = {
    'Agriculture, forestry and fishing': 'A',
    'Mining and quarrying': 'B',
    'Manufacturing': 'C',
    'Electricity, gas, steam and air conditioning supply': 'D',
    'Water supply sewage, waste management and remediation activities': 'E',
    'Construction': 'F',
    'Wholesale and retail trade; repair of motor vehicles and motorcycles': 'G',
    'Transportation and storage': 'H',
    'Accommodation and food service activities': 'I',
    'Information and communication': 'J',
    'Financial and insurance activities': 'K',
    'Real estate activities': 'L',
    'Professional, scientific and technical activities': 'M',
    'Administrative and support service activities': 'N',
    'Public administration and defence; compulsory social security': 'O',
    'Education': 'P',
    'Human health and social work activities': 'Q',
    'Arts, entertainment and recreation': 'R',
    'Other services activities': 'S',
    '': 'Z',
}

NACE_CODE_TO_SEC = {
    'A': 'Agriculture, forestry and fishing',
    'B': 'Mining and quarrying',
    'C': 'Manufacturing',
    'D': 'Electricity, gas, steam and air conditioning supply',
    'E': 'Water supply sewage, waste management and remediation activities',
    'F': 'Construction',
    'G': 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
    'H': 'Transportation and storage',
    'I': 'Accommodation and food service activities',
    'J': 'Information and communication',
    'K': 'Financial and insurance activities',
    'L': 'Real estate activities',
    'M': 'Professional, scientific and technical activities',
    'N': 'Administrative and support service activities',
    'O': 'Public administration and defence; compulsory social security',
    'P': 'Education',
    'Q': 'Human health and social work activities',
    'R': 'Arts, entertainment and recreation',
    'S': 'Other services activities ',
    'Z': 'Other',
}


class EsIsNaFusion:
    """
    Class that combines ESCO, ISCO and NACE knowledge
    """

    def __init__(self, path):
        """
        :param path: path to the folder containing the ESCO data
        """
        self.skill_popularity = {}

        self._isco_tree = Tree(root_id='C')
        self._nace_tree = Tree()
        self._isco_nodes = {}
        self._nace_nodes = {}
        self._skills = {}

        self._occ_en = pd.read_csv(path + 'occupations_en.csv', dtype=str)
        self._skl_en = pd.read_csv(path + 'skills_en.csv', dtype=str)
        self._occ_skl_rel = pd.read_csv(path + 'occupationSkillRelations.csv', dtype=str)
        self._grp_en = pd.read_csv(path + 'ISCOGroups_en.csv', dtype=str)
        self._loc_salary = pd.read_csv(path + 'Salaries.csv', dtype=str)

        self._init_occupations()
        self._init_skills()
        self._link_occ_skl()
        self._init_isco()
        self._init_nace()

        self._stops = set(stopwords.words("english"))
        self._c = CurrencyConverter()

    def _init_occupations(self):
        """
        Initialises the nodes for the ISCO and NACE trees, reading the information from ESCO
        """
        for i in range(len(self._occ_en)):
            name = self._occ_en.preferredLabel[i]
            uri = self._occ_en.conceptUri[i]
            try:
                alts = self._occ_en.altLabels[i].split('\n')
            except AttributeError:
                alts = []
            isco_group = self._occ_en.iscoGroup[i]
            self._isco_nodes[uri] = (Node(ID=isco_group + '.' + name, parent=None,
                                          name=name,
                                          uri=uri,
                                          alts=alts,
                                          essential_skl={'$total_count$': 1},
                                          optional_skl={'$total_count$': 1}))

            self._nace_nodes[uri] = (Node(ID=self._get_nace_code(isco_group) + '.' + name, parent=None,
                                          name=name,
                                          uri=uri,
                                          alts=alts,
                                          essential_skl={'$total_count$': 1},
                                          optional_skl={'$total_count$': 1}))

    def _init_skills(self):
        """
        Initialises the skills dictionary with the data from ESCO
        """
        for i in range(len(self._skl_en)):
            name = self._skl_en.preferredLabel[i]
            uri = self._skl_en.conceptUri[i]
            self._skills[uri] = name

    def _link_occ_skl(self):
        """
        Links the occupation nodes from ISCO and NACE to the skills; compiles an analysis of how popular skills are.
        """
        for i in range(len(self._occ_skl_rel)):
            occupation = self._occ_skl_rel.occupationUri[i]
            skill = self._skills[self._occ_skl_rel.skillUri[i]]
            type = self._occ_skl_rel.relationType[i]
            if type == 'essential':
                self._isco_nodes[occupation].__getattribute__('essential_skl')[skill] = 1
                self._nace_nodes[occupation].__getattribute__('essential_skl')[skill] = 1
            else:
                self._isco_nodes[occupation].__getattribute__('optional_skl')[skill] = 1
                self._nace_nodes[occupation].__getattribute__('optional_skl')[skill] = 1
            if skill in self.skill_popularity:
                self.skill_popularity[skill] += 1
            else:
                self.skill_popularity[skill] = 1

    def _init_isco(self):
        """
        Initialises the ISCO tree and propagates the essential and optional skills through the tree
        """
        for node in self._isco_nodes.values():
            self._isco_tree.add_node(node)
        self._isco_tree.propagate('essential_skl')
        self._isco_tree.propagate('optional_skl')

    def _init_nace(self):
        """
        Initialises the NACE tree and propagates the essential and optional skills through the tree
        """
        for node in self._nace_nodes.values():
            self._nace_tree.add_node(node)
        self._nace_tree.propagate('essential_skl')
        self._nace_tree.propagate('optional_skl')

    @staticmethod
    def _get_nace_code(isco_code):
        """
        Converts an ISCO code to a NACE code.

        :param isco_code: ISCO code to convert
        :type isco_code: str
        :return: equivalent NACE code
        :rtype: str
        """
        nace_sectors = {
            '0': 'Public administration and defence; compulsory social security',
            '111': 'Public administration and defence; compulsory social security',
            '112': 'Professional, scientific and technical activities',
            '12': '',
            '131': 'Agriculture, forestry and fishing',
            '1321': 'Manufacturing',
            '1322': 'Mining and quarrying',
            '1323': 'Construction',
            '1324': 'Transportation and storage',
            '133': 'Information and communication',
            '1341': 'Human health and social work activities',
            '1342': 'Human health and social work activities',
            '1343': 'Human health and social work activities',
            '1344': 'Public administration and defence; compulsory social security',
            '1345': 'Education',
            '1346': 'Financial and insurance activities',
            '1349': 'Professional, scientific and technical activities',
            '141': 'Accommodation and food service activities',
            '142': 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
            '1431': 'Arts, entertainment and recreation',
            '1439': 'Professional, scientific and technical activities',
            '21': 'Professional, scientific and technical activities',
            '22': 'Human health and social work activities',
            '23': 'Education',
            '241': 'Financial and insurance activities',
            '242': 'Administrative and support service activities',
            '243': 'Professional, scientific and technical activities',
            '25': 'Information and communication',
            '261': 'Professional, scientific and technical activities',
            '262': 'Arts, entertainment and recreation',
            '2631': 'Professional, scientific and technical activities',
            '2632': 'Professional, scientific and technical activities',
            '2633': 'Professional, scientific and technical activities',
            '2634': 'Human health and social work activities',
            '2635': 'Human health and social work activities',
            '2636': 'Other services activities',
            '264': 'Information and communication',
            '265': 'Arts, entertainment and recreation',
            '3111': 'Professional, scientific and technical activities',
            '3112': 'Construction',
            '3113': 'Professional, scientific and technical activities',
            '3114': 'Professional, scientific and technical activities',
            '3115': 'Professional, scientific and technical activities',
            '3116': 'Professional, scientific and technical activities',
            '3117': 'Mining and quarrying',
            '3118': 'Professional, scientific and technical activities',
            '3119': 'Professional, scientific and technical activities',
            '3121': 'Mining and quarrying',
            '3122': 'Manufacturing',
            '3123': 'Construction',
            '3131': 'Electricity, gas, steam and air conditioning supply',
            '3132': 'Water supply sewage, waste management and remediation activities',
            '3133': 'Manufacturing',
            '3134': 'Electricity, gas, steam and air conditioning supply',
            '3135': 'Manufacturing',
            '3139': 'Manufacturing',
            '3141': 'Professional, scientific and technical activities',
            '3142': 'Agriculture, forestry and fishing',
            '3143': 'Agriculture, forestry and fishing',
            '315': 'Transportation and storage',
            '321': 'Human health and social work activities',
            '322': 'Human health and social work activities',
            '323': 'Human health and social work activities',
            '324': 'Professional, scientific and technical activities',
            '325': 'Human health and social work activities',
            '331': 'Financial and insurance activities',
            '332': 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
            '333': 'Professional, scientific and technical activities',
            '334': 'Administrative and support service activities',
            '335': 'Public administration and defence; compulsory social security',
            '3411': 'Professional, scientific and technical activities',
            '3412': 'Human health and social work activities',
            '3413': 'Other services activities',
            '342': 'Arts, entertainment and recreation',
            '343': 'Arts, entertainment and recreation',
            '35': 'Information and communication',
            '4': 'Administrative and support service activities',
            '51': 'Other services activities',
            '52': 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
            '53': 'Human health and social work activities',
            '54': 'Public administration and defence; compulsory social security',
            '6': 'Agriculture, forestry and fishing',
            '71': 'Construction',
            '72': 'Manufacturing',
            '73': 'Manufacturing',
            '74': 'Manufacturing',
            '75': 'Manufacturing',
            '811': 'Mining and quarrying',
            '812': 'Manufacturing',
            '813': 'Manufacturing',
            '814': 'Manufacturing',
            '815': 'Manufacturing',
            '816': 'Manufacturing',
            '817': 'Manufacturing',
            '818': 'Manufacturing',
            '82': 'Manufacturing',
            '83': 'Transportation and storage',
            '91': 'Administrative and support service activities',
            '92': 'Agriculture, forestry and fishing',
            '931': 'Mining and quarrying',
            '932': 'Manufacturing',
            '933': 'Transportation and storage',
            '94': 'Accommodation and food service activities',
            '95': 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
            '961': 'Water supply sewage, waste management and remediation activities',
            '962': '',
        }

        for i in range(len(isco_code)):
            code = isco_code[:i + 1]
            if code in nace_sectors:
                return NACE_SEC_TO_CODE[nace_sectors[isco_code[:i + 1]]]
        return ''

    def filter_skills(self, skills):
        nace_vec = []
        isco_vec = []

        for skill in skills:
            n = []
            i = []
            pn = 0
            for node in self._nace_tree.root.children.values():
                tot = node.essential_skl['$total_count$']
                if skill in node.essential_skl:
                    ess = node.essential_skl[skill]
                else:
                    ess = 0
                if skill in node.optional_skl:
                    opt = node.optional_skl[skill]
                else:
                    opt = 0
                probability = (ess + opt) / tot
                pn += probability
                n.append(probability)
            pi = 0
            for node in self._isco_tree.root.children.values():
                tot = node.essential_skl['$total_count$']
                if skill in node.essential_skl:
                    ess = node.essential_skl[skill]
                else:
                    ess = 0
                if skill in node.optional_skl:
                    opt = node.optional_skl[skill]
                else:
                    opt = 0
                probability = (ess + opt) / tot
                pi += probability
                i.append(probability)
            if pn > 0:
                nace_vec.append(np.array(n) / pn)
            else:
                nace_vec.append(np.array(n))
            if pi > 0:
                isco_vec.append(np.array(i) / pi)
            else:
                isco_vec.append(np.array(i))

        torm = []
        for i in range(len(nace_vec)):
            if np.linalg.norm(nace_vec[i]) == 0:
                torm.append(i)
                continue
            if np.linalg.norm(isco_vec[i]) == 0:
                torm.append(i)
                continue

        torm.reverse()

        for i in torm:
            skills.pop(i)
            nace_vec.pop(i)
            isco_vec.pop(i)

        if len(skills) > 2:
            e = 0.1  # 0.1 to 0.5
            nace = []
            for i in range(10):
                clustering = DBSCAN(eps=e, min_samples=2, metric=scipy.spatial.distance.cosine).fit(nace_vec)
                if all(v == 0 for v in clustering.labels_) and e > 0.1:
                    break
                nace.append(clustering.labels_)
                e += 0.1

            e = 0.1  # 0.1 to 0.5
            isco = []
            for i in range(10):
                clustering = DBSCAN(eps=e, min_samples=2, metric=scipy.spatial.distance.cosine).fit(isco_vec)
                if all(v == 0 for v in clustering.labels_) and e > 0.1:
                    break
                isco.append(clustering.labels_)
                e += 0.1

            torm = []
            for i in range(len(skills)):
                if nace[-1][i] == -1 or isco[-1][i] == -1:
                        torm.append(i)

            torm.reverse()
            for i in torm:
                skills.pop(i)

            #n_info = []
            #i_info = []

            #for s in skills:
            #    n_info.append(self._get_information(s, 'nace'))
            #    i_info.append(self._get_information(s, 'isco'))

            #n_support = []
            #i_support = []
            #for i in range(len(nace[0])):
            #    n_support.append(0)
            #    i_support.append(0)
            #    for j in range(len(nace)):
            #        n_cluster = nace[j][i]
            #        if n_cluster != -1:
            #            for k in range(len(nace[0])):
            #                if k != i:
            #                    if nace[j][k] == n_cluster:
            #                        n_support[i] += n_info[k]
            #            # n_support[i] += (nace[j] == n_cluster).sum()
            #        else:
            #            n_support[i] -= n_info[i]
            #        i_cluster = isco[j][i]
            #        if i_cluster != -1:
            #            for k in range(len(nace[0])):
            #                if k != i:
            #                    if isco[j][k] == i_cluster:
            #                        i_support[i] += i_info[k]
            #            # n_support[i] += (nace[j] == n_cluster).sum()
            #        else:
            #            i_support[i] -= i_info[i]

        # get distance to all other vectors
        # distance is multiplied by specificity?

        return skills

    def filter_skills_alt(self, title, skills):
        skillset = set()
        if type(title) == float:
            return []
        title = [i for i in title.lower().split(' ') if i not in self._stops]

        for node in self._isco_nodes.values():
            score = 0
            for w in title:
                if self.findWholeWord(w)(node.name) is not None:
                    score += 1
            score /= max(len(node.name.split(' ')), len(title))
            for a in node.alts:
                alt_score = 0
                for w in title:
                    if self.findWholeWord(w)(a) is not None:
                        alt_score += 1
                alt_score /= max(len(a.split(' ')), len(title))
                if alt_score > score:
                    score = alt_score

            if score > 0:
                for sk in node.essential_skl:
                    skillset.add(sk)
                for sk in node.optional_skl:
                    skillset.add(sk)
        return [i for i in skills if i in skillset]

    def get_sector(self, skills):
        sector = []
        score = []
        for node in self._nace_tree.root.children.values():
            s = 0
            for skill in skills:
                tot = node.essential_skl['$total_count$']
                if skill in node.essential_skl:
                    ess = node.essential_skl[skill]
                else:
                    ess = 0
                if skill in node.optional_skl:
                    opt = node.optional_skl[skill]
                else:
                    opt = 0
                probability = (ess + opt) / tot
                s += probability * self._get_information(skill, 'nace')
            sector.append(node.id)
            score.append(s)

        return NACE_CODE_TO_SEC[sector[score.index(max(score))]]

    def estimate_salary(self, skills, location):
        # 1 get 3 digit ISCO code from skills
        detected = ''
        r_node = self._isco_tree.root
        for i in range(3):
            sector = []
            score = []
            for node in r_node.children.values():
                s = 0
                for skill in skills:
                    tot = node.essential_skl['$total_count$']
                    if skill in node.essential_skl:
                        ess = node.essential_skl[skill]
                    else:
                        ess = 0
                    if skill in node.optional_skl:
                        opt = node.optional_skl[skill]
                    else:
                        opt = 0
                    probability = (ess + opt) / tot
                    s += probability * self._get_information(skill, 'isco')
                sector.append(node.id)
                score.append(s)
            code = sector[score.index(max(score))]
            detected += code
            r_node = r_node.children[code]

        country = pycountry.countries.search_fuzzy(location)[0]
        if country is None:
            print("Country {} not detected".format(location))
            return None
        country_code = country.alpha_2
        # 2 see if we have that in our country, if we do return
        data = self._loc_salary.loc[(self._loc_salary['Country'] == country_code) & (self._loc_salary['ISCO'] == detected)]
        if len(data) == 1:
            amount = float(data.iloc[0][2]) * 12
            currency = pycountry.currencies.get(numeric=country.numeric)
            if currency is not None:
                converted = self._c.convert(amount, currency.alpha_3)
                return int(converted)
            else:
                return int(amount)
        return None
        # 3 iterate through all the other countries and find the closest one based on average wage. Use that country's
        # estimate scaled by the difference in average overrall wage.

    def _get_information(self, skill, tree='isco'):
        if tree == 'isco':
            return 1 - (self._get_entropy(skill, self._isco_tree) / 3.321928094887362)
        elif tree == 'nace':
            return 1 - (self._get_entropy(skill, self._nace_tree) / 4.247927513443583)

    def _get_entropy(self, skill, tree):    # the higher the entropy the less info we get from knowing a job has a skill
        probs = []
        ps = 0
        for node in tree.root.children.values():
            tot = node.essential_skl['$total_count$']
            if skill in node.essential_skl:
                ess = node.essential_skl[skill]
            else:
                ess = 0
            if skill in node.optional_skl:
                opt = node.optional_skl[skill]
            else:
                opt = 0
            probability = (ess + opt) / tot
            ps += probability
            probs.append(probability)
        if ps == 0:
            return 0
        probs = np.array(probs)/ps

        entropy = 0
        for p in probs:
            if p != 0:
                entropy += p * np.log2(p)
        entropy = - entropy
        return entropy

    @staticmethod
    def findWholeWord(w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search



# endregion

# region --- Keyword matching ---


class ESCOMatcher:

    def __init__(self, path, filterPath=None, preprocessor=None, popularity=None):
        data = pd.read_csv(path)
        self._preproc = preprocessor

        if filterPath is not None:
            with open(filterPath, 'r') as f:
                filter_set = f.readlines()
            for i in range(len(filter_set)):
                filter_set[i] = str(filter_set[i]).replace('\n', '').replace('\r', '')
            filter_set = list(set(filter_set))
            filter_set = self._preproc.clean(filter_set, False)

            if '' in filter_set:
                filter_set.remove('')

        preferred = data.preferredLabel.tolist()
        alternative = data.altLabels.tolist()
        for i in range(len(alternative)):
            if type(alternative[i]) == float:
                alternative[i] = ''
            else:
                alternative[i] = alternative[i].replace('\n', ' s p l i t ') + ' s p l i t ' + preferred[i]
        alternative = self._preproc.clean(alternative, False)
        self._labels = {}
        # with tqdm(total=len(preferred), unit=' matches') as pbar:
        for i in range(len(preferred)):
            if preferred[i] == 'electricity principles' or preferred[i] == 'design ventilation network' or preferred[
                i] == 'perform ground-handling maintenance procedures' or preferred[
                i] == 'compile airport certification manuals':
                continue
            self._labels[preferred[i]] = [preferred[i]]
            if alternative[i] != '':
                for lbl in alternative[i].split(' s p l i t '):
                    if filterPath is not None and lbl not in filter_set:
                        continue
                    if lbl == 'reporting' or \
                            lbl == 'it' or \
                            lbl == 'interview' or \
                            lbl == 'access' or \
                            lbl == 'history' or \
                            lbl == 'security' or \
                            lbl == 'balance' or \
                            lbl == 'energy' or \
                            lbl == 'engineering':
                        continue
                    if lbl in self._labels.keys():
                        self._labels[lbl].append(preferred[i])
                    else:
                        self._labels[lbl] = [preferred[i]]
            # pbar.update()

        for key in self._labels:
            vals = self._labels[key]
            pop = None
            count = -1
            if len(vals) > 1:
                k = 0
            for v in vals:
                if v in popularity:
                    popv = popularity[v]
                else:
                    popv = 0
                if popv > count:
                    pop = v
                    count = popv
            self._labels[key] = pop

        self._create_automaton()

    def match_ESCO(self, text, duplicates):
        matches = []
        kw_matches = []
        with tqdm(total=len(text), unit=' matches') as pbar:
            for ad in text:
                jobs = []
                start = 0
                for end, job in self._match(ad):
                    if self._find_word(job[1])(ad[start:]) is not None:
                        jobs.append(job[1])
                        start = end + 1
                if not duplicates:
                    jobs = list(set(jobs))
                kws = []
                for j in jobs:
                    kws.append(self._labels[j])
                if not duplicates:
                    kws = list(set(kws))
                matches.append(jobs)
                kw_matches.append(kws)
                pbar.update()
        return matches, kw_matches

    def _create_automaton(self):
        keywords = set(self._labels.keys())
        self._automaton = Automaton()
        i = 0
        for k in keywords:
            self._automaton.add_word(k, (i, k))
            i += 1
        self._automaton.make_automaton()

    def _match(self, text):
        return self._automaton.iter(text)

    @staticmethod
    def _find_word(w):
        return re.compile(r'\b(' + re.escape(w) + r')\b').search


class FilterMatcher:

    def __init__(self, model_path, keywords_path, preprocessor=None):
        self._matcher = Matcher(keywords_path, preprocessor)
        self._model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def match(self, text):
        filtered_keywords = []
        with tqdm(total=len(text), unit=' matches') as pbar:
            for posting in text:
                if isinstance(posting, float):
                    filtered_keywords.append('')
                    continue
                posting = posting.lower()
                match = []
                start = 0
                for end, k in self._matcher.match(posting):
                    if self._matcher.find_word(k[1])(posting[start:]) is not None:
                        match.append(k[1])
                        start = end + 1

                vectors = []
                torm = []
                for m in match:
                    m1 = m.replace(" ", "_")
                    if m1 in self._model.vocab:
                        vectors.append(self._model[m1])
                        continue
                    else:
                        m1 = m1.split("_")
                    try:
                        mv = self._model[m1[0]]
                    except KeyError:
                        torm.append(m)
                        continue
                    for i in range(len(m1) - 1):
                        try:
                            mv = np.add(mv, self._model[m1[i + 1]])
                        except KeyError:
                            continue
                    vectors.append(mv)
                for item in torm:
                    match.remove(item)
                while len(vectors) > 2:
                    to_rm = furthest(vectors)
                    vectors = np.delete(vectors, to_rm, 0)
                    match.remove(match[to_rm])

                # match = list(set(match))

                if len(match) == 2:
                    if match[0] == match[1]:
                        match = match[0]
                    elif match[0] in match[1]:
                        match = match[1]
                    elif match[1] in match[0]:
                        match = match[0]
                    else:
                        match = match[0]
                elif len(match) == 1:
                    match = match[0]
                if len(match) > 0:
                    filtered_keywords.append(match.title())
                else:
                    filtered_keywords.append('')
                pbar.update()

        return filtered_keywords


class Matcher:

    def __init__(self, keywords_path, preprocessor=None):
        self._kp = keywords_path
        self._preprocessor = preprocessor
        self._create_automaton()

    def match(self, text):
        return self._automaton.iter(text)

    def find_word(self, w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    def _create_automaton(self):
        with open(self._kp, 'rb') as f:
            keywords = f.readlines()
        for i in range(len(keywords)):
            keywords[i] = str(keywords[i])[2:-3]
        if self._preprocessor is not None:
            keywords = self._preprocessor.clean(keywords, False)
        keywords = set(keywords)
        if "" in keywords:
            keywords.remove("")

        self._automaton = Automaton()
        i = 0
        for k in keywords:
            self._automaton.add_word(k, (i, k))
            i += 1
        self._automaton.make_automaton()


def furthest(vectors):
    distances = []
    for v1 in vectors:
        d = 0
        for v2 in vectors:
            distance = np.abs(spatial.distance.cosine(v1, v2))
            d += distance
        distances.append(d)
    return np.argmax(distances)

# endregion
