from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from mad.models.space import distance_model
from mad.models.uq import ensemble_model
from mad.ml.assessment import NestedCV
from mad.datasets import load_data
from mad.ml import splitters


def main():

    # Load data
    data = load_data.diffusion(frac=1)
    df = data['frame']
    X = data['data']
    y = data['target']
    g = data['class_name']

    # ML Distance model
    ds_model = distance_model(dist='gpr_std')

    # ML UQ function
    uq_model = ensemble_model()

    # ML Pipeline
    scale = StandardScaler()
    model = RandomForestRegressor()
    selector = SelectFromModel(model)

    grid = {}
    grid['model__n_estimators'] = [100]
    grid['model__max_features'] = [None]
    grid['model__max_depth'] = [None]
    pipe = Pipeline(steps=[
                           ('scaler', scale),
                           #('select', selector),
                           ('model', model)
                           ])
    gs_model = GridSearchCV(pipe, grid, cv=RepeatedKFold(n_repeats=1))

    # Types of sampling to test
    splits = [('random', RepeatedKFold(n_repeats=1))]

    # Chemical splits
    n_groups = len(set(g))
    if n_groups > 1:
        chem_split = ('chemical', splitters.LeaveOneGroupOut())
        splits.append(chem_split)

    for i in [2, 4]:

        # Cluster Splits
        top_split = splitters.RepeatedClusterSplit(
                                                   KMeans,
                                                   n_repeats=1,
                                                   n_clusters=i
                                                   )

        splits.append(('kmeans_{}'.format(i), top_split))

    for i in splits:
        spl = NestedCV(X, y, g, i[1])
        spl.assess(gs_model, uq_model, ds_model, save='runs/'+i[0])
        spl.save_model(gs_model, uq_model, ds_model, save='runs/'+i[0])

if __name__ == '__main__':
    main()

