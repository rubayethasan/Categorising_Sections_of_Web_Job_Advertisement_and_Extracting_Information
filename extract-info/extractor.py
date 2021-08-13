import time

import pandas as pd

from tqdm import tqdm
from utilities import FilterMatcher, Preprocessor, EsIsNaFusion, ESCOMatcher


class Extractor:

    def __init__(self, params):
        self._params = params
        self._features = {}
        self._groups = EsIsNaFusion(self._params['downloads_path'] + '/v1.0.8/')
        self._preprocessor = Preprocessor(mode=params['preprocessor_mode'])

    def extract(self, ids, text, urls, display):
        self._features['id'] = ids
        self._features['url'] = urls

        if not {'job_title', 'skills', 'sector', 'estimated_salary'}.issubset(set(self._params['to_calculate'])):
            df = pd.read_csv(self._params['out_path_a'])
            if self._params['num_postings'] != 0:
                df = df.head(self._params['num_postings'])
        if self._calc(['job_title', 'skills']):
            display.set_state(22)
            self._features['ad'] = self._preprocessor.clean(text)
        #else:
        #    self._features['ad'] = df['ad'].tolist()
        if self._calc(['job_title']):
            display.set_state(23)
            self._features['job_title'] = self._get_title()
        else:
            self._features['job_title'] = df['job_title'].tolist()
        if self._calc(['skills']):
            self._features['skills'] = self._get_skills(display)
        else:
            self._features['skills'] = df['skills'].tolist()
            self._fsk = []
            for sl in df['skills'].tolist():
                if type(sl) == float:
                    self._fsk.append([])
                    continue
                sl = sl.lower().split('; ')
                self._fsk.append(sl)
        if self._calc(['sector']):
            display.set_state(26)
            self._features['sector'] = self._get_sector()
        else:
            self._features['sector'] = df['sector'].tolist()
        if self._calc(['estimated_salary']):
            self._features['estimated_salary'] = self._estimate_salary()
        else:
            self._features['estimated_salary'] = df['estimated_salary'].tolist()

        self._features.pop('ad', None)

        dataframe = pd.DataFrame.from_dict(self._features)
        self._features = {}
        return dataframe

    def _get_title(self):
        matcher = FilterMatcher(self._params['downloads_path'] + '/GoogleNews-vectors-negative300.bin',
                                self._params['downloads_path'] + '/jobs.txt',
                                self._preprocessor)
        matches = matcher.match(self._features['ad'])
        return matches

    def _get_skills(self, display):
        display.set_state(24)
        matcher = ESCOMatcher(self._params['downloads_path'] + '/v1.0.8/skills_en.csv', None,  # 'data/linkedin_skill.txt',
                              self._preprocessor,
                              self._groups.skill_popularity)
        m, km = matcher.match_ESCO(self._features['ad'], False)

        display.set_state(25)
        filtered = []
        with tqdm(total=len(km), unit=' matches') as pbar:
            for i in range(len(km)):
                filtered.append(self._groups.filter_skills_alt(self._features['job_title'][i], km[i][:]))
                pbar.update()

        self._fsk = filtered

        res = []
        for i in range(len(filtered)):
            s = ''
            for skill in filtered[i]:
                s += skill.capitalize() + '; '
            s = s[:-2]
            res.append(s)

        return res

    def _get_sector(self):
        sectors = []
        with tqdm(total=len(self._fsk), unit=' matches') as pbar:
            for skills in self._fsk:
                if len(skills) == 0:
                    sectors.append('')
                    pbar.update()
                    continue
                # code = self._groups.groups.get_skill_path(skills)
                # sectors.append(self._groups.groups.get_sector(code))
                sectors.append(self._groups.get_sector(skills))
                pbar.update()
        return sectors

    def _estimate_salary(self):
        salaries = []
        locations = list(pd.read_csv(self._params['out_path_r']).job_location)
        with tqdm(total=len(self._fsk), unit=' matches') as pbar:
            for i in range(len(self._fsk)):
                skills = self._fsk[i]
                loc = locations[i]
                if type(loc) != str or loc is None:
                    salaries.append(None)
                    continue
                loc = loc.split(',')[0]
                if len(skills) == 0 and len(loc):
                    salaries.append(None)
                    pbar.update()
                    continue
                # code = self._groups.groups.get_skill_path(skills)
                # sectors.append(self._groups.groups.get_sector(code))
                salaries.append(self._groups.estimate_salary(skills, loc))
                pbar.update()
        return salaries

    def _calc(self, features):
        return set(features) & set(self._params['to_calculate'])


def run_extraction(params, display):
    display.set_state(21)
    df = pd.read_csv(params['in_path_a'])
    if params['num_postings'] != 0:
        df = df.head(params['num_postings'])
    ids = list(df.id)
    text = list(df.job_desc_plain)
    urls = list(df.url)
    e = Extractor(params)
    extracted_data = e.extract(ids, text, urls, display)
    extracted_data.to_csv(params['out_path_a'], index=False)
    display.set_state(0)


if __name__ == '__main__':
    parameters = {
        'in_path':           '../data/parsed-data/euro-jobs-parsed.csv',
        'out_path':          '../data/extracted-data/merged/result-andr.csv',
        'downloads_path':    '../data/downloads',
        'preprocessor_mode': 2,
    }
    # run_extraction(parameters, )
