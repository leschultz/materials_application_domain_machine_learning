from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import RepeatedKFold
from mad.datasets import load_data
from mad.ml.assessment import NestedCV
from mad.models.space import distance_model
from mad.models.uq import ensemble_model


def main():

    # Load data
    data = load_data.diffusion(frac=1)
    df = data['frame']
    X = data['data']
    y = data['target']
    g = data['class_name']

    # ML Splits    
    splitter = RepeatedKFold(n_repeats=2)

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
    gs_model = GridSearchCV(pipe, grid, cv=RepeatedKFold(n_repeats=10))

    spl = NestedCV(X, y, g, splitter)
    spl.assess(gs_model, uq_model, ds_model, save='random')
    spl.save_model(gs_model, uq_model, ds_model, save='random')

if __name__ == '__main__':
    main()

