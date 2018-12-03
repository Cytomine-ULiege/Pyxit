# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2018.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

__author__ = "Mormont Romain <r.mormont@ulg.ac.be>"
__copyright__ = "Copyright 2010-2018 University of Liège, Belgium, http://www.cytomine.be/"

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
    n_jobs = base_estimator_params.get("n_jobs", 1)
    random_state = check_random_state(random_state)
    et = ExtraTreesClassifier(random_state=random_state, **base_estimator_params)
    pyxit = PyxitClassifier(
        base_estimator=et,
        n_subwindows=n_subwindows,
        min_size=min_size,
        max_size=max_size,
        target_height=target_height,
        target_width=target_width,
        n_jobs=n_jobs,
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
