# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.ackley import Ackley
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.dixon_price import DixonPrice
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.griewank import Griewank
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.langermann import Langermann
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.levy import Levy
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.michalewicz import Michalewicz
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.modified_sphere import ModifiedSphere
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.perm import Perm
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.perm0 import Perm0
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.powell import Powell
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.power_sum import PowSum
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.rastrigin import Rastrigin
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.rosenbrock import Rosenbrock
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.rotated_hyper_ellipsoid import RotHyp
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.schwefel import Schwefel
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.sphere import Sphere
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.styblinski_tang import StyblinskiTang
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.sum_pow import SumPow
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.sum_squares import SumSquares
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.trid import Trid
from HEBO.MCBO.mcbo.tasks.synthetic.sfu.sfu_functions.zakharov import Zakharov

MANY_LOCAL_MINIMA = {'ackley': Ackley,
                     'griewank': Griewank,
                     'langermann': Langermann,
                     'levy': Levy,
                     'rastrigin': Rastrigin,
                     'schwefel': Schwefel}

BOWL_SHAPED = {'perm0': Perm0,
               'rot_hyp': RotHyp,
               'sphere': Sphere,
               'modified_sphere': ModifiedSphere,
               'sum_pow': SumPow,
               'sum_squares': SumSquares,
               'trid': Trid}

PLATE_SHAPED = {'power_sum': PowSum,
                'zakharov': Zakharov}

VALLEY_SHAPED = {'dixon_price': DixonPrice,
                 'rosenbrock': Rosenbrock}

STEEP_RIDGES = {'michalewicz': Michalewicz}

OTHER = {'perm': Perm,
         'powell': Powell,
         'styblinski_tang': StyblinskiTang}
