from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

from pyxit.estimator import _get_output_from_directory, PyxitClassifier, SvmPyxitClassifier


def build_models(n_subwindows=10, min_size=0.5, max_size=1.0, target_width=16, target_height=16, interpolation=2,
                 transpose=False, colorspace=2, fixed_size=False, verbose=0, get_output=_get_output_from_directory,
                 create_svm=False, C=1.0, random_state=None, **base_estimator_params):
    """Build models
    Parameters
    ----------
    n_subwindows: int
    min_size: float
    max_size: float
    target_width: int
    target_height: int
    interpolation: int
    transpose: bool
    colorspace: int
    fixed_size: bool
    verbose: int
    get_output: callable
    create_svm: bool
    C: float
    base_estimator_params: dict
        Parameters for the ExtraTreesClassifier object

    Returns
    -------
    et: ExtraTreesClassifier
        Base estimator a.k.a. extra-trees
    pyxit: PyxitClassifier|SvmPyxitClassifier
        (Svm) Pyxit classifier
    """
    # make sure n_jobs is only used for parallelizing at the PyxitClassifier level (data parallelism)
    base_estimator_params.pop("n_jobs")
    random_state = check_random_state(random_state)
    et = ExtraTreesClassifier(random_state=random_state, **base_estimator_params)
    pyxit = PyxitClassifier(
        base_estimator=et,
        n_subwindows=n_subwindows,
        min_size=min_size,
        max_size=max_size,
        target_height=target_height,
        target_width=target_width,
        n_jobs=1,  # n_jobs is used to
        colorspace=colorspace,
        fixed_size=fixed_size,
        interpolation=interpolation,
        transpose=transpose,
        verbose=verbose,
        get_output=get_output,
        random_state=check_random_state(random_state.tomaxint() % 0x100000000)  # ET and Pyxit must have != random nbs
    )
    if not create_svm:
        return et, pyxit
    else:
        return et, SvmPyxitClassifier(pyxit, LinearSVC(C=C))
