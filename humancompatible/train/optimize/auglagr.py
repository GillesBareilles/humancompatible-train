import torch.utils.data.dataloader
import torch.optim as optim
import numpy as np
from .stochastic_constraint import StochasticConstraint
from typing import Callable

# def SampleFromDataloader(d: torch.utils.data.DataLoader):
#     try:
#         c_inputs, c_labels = next(constr_data_iterator)
#     except StopIteration:
#         constr_data_iterator = iter(d)
#         c_inputs, c_labels = next(constr_data_iterator)

class ALOptimizer:
    def __init__(self,
                 net: torch.nn.Module,
                 loss_fn: Callable,
                 m_det: int,
                 m_st: int,
                 lambda_0: torch.Tensor=None,
                 aug_term: float=1.,
                 t: float=1.01,
                 loss_grad: Callable=None,
                 det_constr_grad: Callable=None,
                 st_constr_grad: Callable=None,
                 det_constraint_fn: Callable=None,
                 st_constraint_fn: Callable | StochasticConstraint | None =None,
                 minimizer: str='Adam',
                 **kwargs):
        """
        Solves an inequality-constrained optimization problem using the non-smooth Augmented Lagrangian method:
        min f(x) s.t. c(x) <= 0

        Args:
            net (torch.nn.Module): the network to train.
            # params (iterable): iterable of parameters to optimize.
            loss_fn (Callable): the objective function.
            m_det (int): number of deterministic constraints.
            m_st (int): number of stochastic constraints.
            lambda_0 (torch.Tensor, optional): initial value for the Lagrange multipliers, expected to be of shape (m,).
            aug_term (float): initial value for the r (rho) parameter.
            t (float): multiplier for the r (rho) parameter.
            det_constraint_fn (Callable, optional): the deterministic constraint function, expected: f(net) -> Tensor((m,)); 
            expected to return a tensor of constraint values.
            st_constraint_fn (Callable, optional): the stochastic constraint function, expected:
            f(net, inputs, labels) -> Tensor((m,)); expected to return a tensor of constraint values.
            minimizer (str, optional): the PyTorch optimizer to minimize the objective function on each iteration; default is Adam.
            **kwargs: arguments for the PyTorch optimizer.
        """
        # self._params = params
        self.net = net
        
        self._loss_fn = loss_fn
        self._det_constraint_fn = lambda net: torch.max(det_constraint_fn(net),
                                                        torch.zeros(m_det))
        
        if isinstance(st_constraint_fn, Callable): 
            self._st_constraint_fn = lambda net, data: torch.max(st_constraint_fn(net, data),
                                                       torch.zeros(m_st))
        elif isinstance(st_constraint_fn, StochasticConstraint):
            self.st_constraint = st_constraint_fn
            self._st_constraint_fn = self.st_constraint.fun

        self.m_det = m_det
        self.m_st = m_st
        self._lambda = lambda_0 if lambda_0 is not None else torch.ones(self.m_det + self.m_st)
        self.__start_lambda = self._lambda
        self.__start_lambda.copy_(self._lambda)
        self._ss = aug_term
        self.__start_ss = aug_term
        self._t = t

        if minimizer is None or minimizer.lower() == 'adam':
            self._optimizer = optim.Adam(self.net.parameters(), **kwargs)
        elif minimizer.lower() == 'sgd':
            self._optimizer = optim.SGD(self.net.parameters(), **kwargs)
        else:
            raise ValueError(f'Unknown optimizer: {minimizer}')

    # def optimize(self,
    #              data: torch.utils.data.DataLoader,
    #              con_data: torch.utils.data.DataLoader=None,
    #              constr_sampling_interval: int | str = 1,
    #              maxiter: int=3,
    #              epochs: int=3,
    #              verbose: bool=True) -> None:
    #     """
    #     Perform optimization using the Augmented Lagrangian method.

    #     Iteratively minimize the augmented Lagrangian function with respect to 
    #     the neural network parameters while updating the Lagrange multipliers and augmentation term.
    #     The method updates the network parameters in place and records optimization history.

    #     Args:
    #         dataset (torch.utils.data.DataLoader): DataLoader providing input data and labels for training.
    #         constr_data (torch.utils.data.DataLoader): iterable-type DataLoader providing input data and labels
    #         for evaluating stochastic constraints.
    #         constr_sampling_interval (int | str, optional): number of algorithm iterations between resampling data for 
    #         stochastic constraint evaluation. 
    #         epochs (int, optional): Number of epochs. Default is 3.
    #         maxiter (int, optional): Number of iterations for updating the Lagrange multipliers per epoch. Default is 3.
    #         verbose (bool, optional): Whether to print progress and constraint updates. Default is True.

    #     Returns:
    #         None 
    #     """
        
    #     if con_data is None and self.m_st > 0:
    #         raise ValueError('constr_data missing even though m_st > 0')
        
    #     self.loss_val = 0
    #     self.history = {'L': [], 'loss': [], 'constr': []}
        
    #     if self.m_st > 0:
    #         constr_data_iterator = con_data.__iter__()

    #     if constr_sampling_interval == 'exp':
    #         last_sampling = 0

    #     for epoch in range(epochs):
    #         # we update Lagrange multipliers up to maxiter times
    #         for lag_iter in range(maxiter):
    #             # check if need to resample this iteration
    #             if self.m_st > 0 and (
    #                     (constr_sampling_interval == 'exp' and (lag_iter == last_sampling ** 2 or lag_iter <= 2))
    #                     or 
    #                     (constr_sampling_interval != 'exp' and lag_iter % constr_sampling_interval == 0)
    #                 ):
    #                 # sample data for constraint evaluation
    #                 try:
    #                     c_inputs, c_labels = next(constr_data_iterator)
    #                 except StopIteration:
    #                     constr_data_iterator = iter(con_data)
    #                     c_inputs, c_labels = next(constr_data_iterator)
    #                 last_sampling = lag_iter
    #             # minimize the Augmented Lagrangian for given aug.lag. multiplier values
    #             # (we recalculate loss and constraint values each iteration here) 
    #             for i, data in enumerate(data):
    #                 self._optimizer.zero_grad()

    #                 # evaluate constraints
    #                 constraint_eval = []
    #                 if self.m_det > 0:
    #                     det_constraint_eval = self._det_constraint_fn(self.net)
    #                     if det_constraint_eval.ndim == 0:
    #                         det_constraint_eval = det_constraint_eval.reshape(1)
    #                     constraint_eval.append(det_constraint_eval)
    #                 if self.m_st > 0:
    #                     st_constraint_eval = self._st_constraint_fn(self.net, c_inputs, c_labels)
    #                     if st_constraint_eval.ndim == 0:
    #                         st_constraint_eval = st_constraint_eval.reshape(1)
    #                     constraint_eval.append(st_constraint_eval)
    #                 # if self.m_st == 0 and self.m_det == 0:
    #                 constraint_eval = torch.concatenate(constraint_eval)
                    
    #                 # evaluate loss
    #                 inputs, labels = data
    #                 outputs = self.net(inputs)
    #                 loss_eval = self._loss_fn(outputs, labels)
                    
    #                 # evaluate augmented lagrangian
    #                 L = loss_eval + self._lambda @ constraint_eval + 0.5*self._ss*torch.sum(torch.square(constraint_eval))
                    
    #                 # pytorch optimizer step
    #                 L.backward()
    #                 self._optimizer.step()

    #                 ###
    #                 if verbose:
    #                     print(f'{epoch}, {lag_iter}, {i}, {loss_eval.detach().item()}, {[v.detach().item() for v in constraint_eval]}', end='\r')
    #                     # print("\r" , end="")
    #                 self.history['L'].append(L)
    #                 self.history['loss'].append(loss_eval)
    #                 self.history['constr'].append(constraint_eval)
    #                 ###
            
    #             # update Lagrange multipliers after minimizing the Augmented Lagrangian
    #             with torch.no_grad():
    #                 # evaluate constraints
    #                 constraint_eval = []
    #                 if self.m_det > 0:
    #                     det_constraint_eval = self._det_constraint_fn(self.net)
    #                     if det_constraint_eval.ndim == 0:
    #                         det_constraint_eval = det_constraint_eval.reshape(1)
    #                     constraint_eval.append(det_constraint_eval)
    #                 if self.m_st > 0:
    #                     st_constraint_eval = self._st_constraint_fn(self.net, c_inputs, c_labels)
    #                     if st_constraint_eval.ndim == 0:
    #                         st_constraint_eval = st_constraint_eval.reshape(1)
    #                     constraint_eval.append(st_constraint_eval)
    #                 constraint_eval = torch.concatenate(constraint_eval)
                    
    #                 if verbose:
    #                     # print('-------')
    #                     print('\n')
    #                 self._lambda += self._ss*constraint_eval
    #                 self._ss *= self._t

        #######################
        # stopping condition? #
        #######################

    def optimize_(self,
                 train_data: torch.utils.data.DataLoader,
                 con_data: torch.utils.data.DataLoader=None | list[torch.utils.data.DataLoader],
                 constr_sampling_interval: int | str = 1,
                 maxiter: int=3,
                 epochs: int=3,
                 verbose: bool=True) -> None:
        """
        Perform optimization using the Augmented Lagrangian method.

        Iteratively minimize the augmented Lagrangian function with respect to 
        the neural network parameters while updating the Lagrange multipliers and augmentation term.
        The method updates the network parameters in place and records optimization history.

        Args:
            dataset (torch.utils.data.DataLoader): DataLoader providing input data and labels for training.
            constr_data (torch.utils.data.DataLoader): iterable-type DataLoader providing input data and labels
            for evaluating stochastic constraints.
            constr_sampling_interval (int | str, optional): number of algorithm iterations between resampling data for 
            stochastic constraint evaluation. 
            epochs (int, optional): Number of epochs. Default is 3.
            maxiter (int, optional): Number of iterations for updating the Lagrange multipliers per epoch. Default is 3.
            verbose (bool, optional): Whether to print progress and constraint updates. Default is True.

        Returns:
            None 
        """
        
        if con_data is None and self.m_st > 0:
            raise ValueError('constr_data missing even though m_st > 0')
        
        self.loss_val = 0
        self.history = {'L': [], 'loss': [], 'constr': []}
        
        if self.m_st > 0:
            constr_data_iterator = con_data.__iter__() if not isinstance(con_data, list) else zip(*[iter(c) for c in con_data])

        if constr_sampling_interval == 'exp':
            last_sampling = 0

        for epoch in range(epochs):
            # we update Lagrange multipliers up to maxiter times
            for lag_iter in range(maxiter):
                # check if need to resample this iteration
                if self.m_st > 0 and (
                        (constr_sampling_interval == 'exp' and (lag_iter == last_sampling ** 2 or lag_iter <= 2))
                        or 
                        (constr_sampling_interval != 'exp' and lag_iter % constr_sampling_interval == 0)
                    ):
                    # sample data for constraint evaluation
                    try:
                        # if one data_loader, will be (inputs (shapeXbatch_size), labels (batch_size))
                        # if more, will be [(inputs1 (shapeXbatch_size), labels1 (batch_size)), (inputs2 (shapeXbatch_size), labels2 (batch_size)), ...]
                        c_data = next(constr_data_iterator)
                    except StopIteration:
                        constr_data_iterator = con_data.__iter__() if not isinstance(con_data, list) else zip(*[iter(c) for c in con_data])
                        c_data = next(constr_data_iterator)
                    last_sampling = lag_iter
                # minimize the Augmented Lagrangian for given aug.lag. multiplier values
                # (we recalculate loss and constraint values each iteration here) 
                for i, data in enumerate(train_data):
                    self._optimizer.zero_grad()

                    # evaluate constraints
                    constraint_eval = []
                    if self.m_det > 0:
                        det_constraint_eval = self._det_constraint_fn(self.net)
                        if det_constraint_eval.ndim == 0:
                            det_constraint_eval = det_constraint_eval.reshape(1)
                        constraint_eval.append(det_constraint_eval)
                    if self.m_st > 0:
                        st_constraint_eval = self._st_constraint_fn(self.net, c_data)
                        if st_constraint_eval.ndim == 0:
                            st_constraint_eval = st_constraint_eval.reshape(1)
                        constraint_eval.append(st_constraint_eval)
                    # if self.m_st == 0 and self.m_det == 0:
                    constraint_eval = torch.concatenate(constraint_eval)
                    
                    # evaluate loss
                    inputs, labels = data
                    outputs = self.net(inputs)
                    loss_eval = self._loss_fn(outputs, labels)
                    
                    # evaluate augmented lagrangian
                    L = loss_eval + self._lambda @ constraint_eval + 0.5*self._ss*torch.sum(torch.square(constraint_eval))
                    
                    # pytorch optimizer step
                    L.backward()
                    self._optimizer.step()

                    ###
                    if verbose:
                        print(epoch, lag_iter, i,
                              '{:.10f}'.format(loss_eval.detach().item()),
                              ['{:10.8f}'.format(v.detach().item()) for v in constraint_eval],
                              end='\r')
                    self.history['L'].append(L.detach().item())
                    self.history['loss'].append(loss_eval.detach().item())
                    self.history['constr'].append(constraint_eval.detach().item())
                    ###
            
                # update Lagrange multipliers after minimizing the Augmented Lagrangian
                with torch.no_grad():
                    # evaluate constraints
                    constraint_eval = []
                    if self.m_det > 0:
                        det_constraint_eval = self._det_constraint_fn(self.net)
                        if det_constraint_eval.ndim == 0:
                            det_constraint_eval = det_constraint_eval.reshape(1)
                        constraint_eval.append(det_constraint_eval)
                    if self.m_st > 0:
                        st_constraint_eval = self._st_constraint_fn(self.net, c_data)
                        if st_constraint_eval.ndim == 0:
                            st_constraint_eval = st_constraint_eval.reshape(1)
                        constraint_eval.append(st_constraint_eval)
                    constraint_eval = torch.concatenate(constraint_eval)
                    
                    if verbose:
                        # print('-------')
                        print('\n')
                    self._lambda += self._ss*constraint_eval
                    self._ss *= self._t

    def optimize_class(self,
                 train_data: torch.utils.data.DataLoader,
                 constr_sampling_interval: int | str = 1,
                 maxiter: int=3,
                 epochs: int=3,
                 verbose: bool=True) -> None:
        """
        Perform optimization using the Augmented Lagrangian method.

        Iteratively minimize the augmented Lagrangian function with respect to 
        the neural network parameters while updating the Lagrange multipliers and augmentation term.
        The method updates the network parameters in place and records optimization history.

        Args:
            dataset (torch.utils.data.DataLoader): DataLoader providing input data and labels for training.
            constr_sampling_interval (int | str, optional): number of algorithm iterations between resampling data for 
            stochastic constraint evaluation. 
            epochs (int, optional): Number of epochs. Default is 3.
            maxiter (int, optional): Number of iterations for updating the Lagrange multipliers per epoch. Default is 3.
            verbose (bool, optional): Whether to print progress and constraint updates. Default is True.

        Returns:
            None 
        """
        
        self.loss_val = 0
        self.history = {'L': [], 'loss': [], 'constr': []}
        old_constraint_eval = np.inf
        if constr_sampling_interval == 'exp':
            last_sampling = 0

        for epoch in range(epochs):
            # we update Lagrange multipliers up to maxiter times
            for lag_iter in range(maxiter):
                # check if need to resample this iteration
                if self.m_st > 0 and (
                        (constr_sampling_interval == 'exp' and (lag_iter == last_sampling ** 2 or lag_iter <= 2))
                        or 
                        (constr_sampling_interval != 'exp' and lag_iter % constr_sampling_interval == 0)
                    ):
                    # sample data for constraint evaluation
                    # TODO: we have multiple constraints
                    c_data = self.st_constraint.sample()
                    last_sampling = lag_iter
                # minimize the Augmented Lagrangian for given aug.lag. multiplier values
                # (we recalculate loss and constraint values each iteration here) 
                for i, data in enumerate(train_data):
                    self._optimizer.zero_grad()

                    # evaluate constraints
                    constraint_eval = self.eval_constraints(c_data)
                    
                    # evaluate loss
                    inputs, labels = data
                    outputs = self.net(inputs)
                    loss_eval = self._loss_fn(outputs, labels)
                    
                    # evaluate augmented lagrangian
                    L = loss_eval + self._lambda @ constraint_eval + 0.5*self._ss*torch.sum(torch.square(constraint_eval))
                    
                    # pytorch optimizer step
                    L.backward()
                    self._optimizer.step()

                    ###
                    if verbose:
                        print(epoch, lag_iter, i,
                              '{:.10f}'.format(loss_eval.detach().item()),
                              '{:.10f}'.format(L.detach().item()),
                              ['{:.10f}'.format(v.detach().item()) for v in constraint_eval],
                              end='\r')
                    self.history['L'].append(L.detach().item())
                    self.history['loss'].append(loss_eval.detach().item())
                    self.history['constr'].append(constraint_eval.detach().item())
                    ###
            
                # update Lagrange multipliers after minimizing the Augmented Lagrangian
                with torch.no_grad():
                    # evaluate constraints
                    constraint_eval = self.eval_constraints(c_data)

                    if verbose:
                        # print('-------')
                        print('\n')
                    self._lambda += self._ss*constraint_eval
                    if constraint_eval > 0:
                    # if constraint_eval/old_constraint_eval > 0.25:
                        self._ss *= self._t
                    old_constraint_eval = constraint_eval


    def eval_constraints(self, st_sample):
        constraint_eval = []
        if self.m_det > 0:
            det_constraint_eval = self._det_constraint_fn(self.net)
            if det_constraint_eval.ndim == 0:
                det_constraint_eval = det_constraint_eval.reshape(1)
            constraint_eval.append(det_constraint_eval)
        if self.m_st > 0:
            # TODO: collect evals of each constraint into a tensor, then do max
            st_constraint_eval = self._st_constraint_fn(self.net, st_sample)
            st_constraint_eval = torch.max(st_constraint_eval, torch.zeros(1))
            if st_constraint_eval.ndim == 0:
                st_constraint_eval = st_constraint_eval.reshape(1)
            constraint_eval.append(st_constraint_eval)
        # if self.m_st == 0 and self.m_det == 0:
        constraint_eval = torch.concatenate(constraint_eval)

        return constraint_eval


    # def optimize_cond(self, dataloader: torch.utils.data.DataLoader, maxiter: int=3, epochs: int=3, verbose: bool=True,
    #                   con_decrease_tol: float = 2, early_stopping: int = 3, con_stopping_tol: float=1e-3) -> None:
    #     """
    #     Perform optimization using the Augmented Lagrangian method.

    #     Iteratively minimize the augmented Lagrangian function with respect to 
    #     the neural network parameters while updating the Lagrange multipliers and augmentation term.
    #     The method updates the network parameters in place and records optimization history.

    #     Args:
    #         dataloader (torch.utils.data.DataLoader): DataLoader providing input data and labels for training.
    #         epochs (int, optional): Number of epochs. Default is 3.
    #         maxiter (int, optional): Number of iterations for updating the Lagrange multipliers per epoch. Default is 3.
    #         verbose (bool, optional): Whether to print progress and constraint updates. Default is True.

    #     Returns:
    #         None 
    #     """

    #     _prev_constr = np.inf
    #     _total_best_loss = np.inf
    #     _no_loss_improvement_epochs = 0

    #     self.loss_val = 0
    #     self.history = {'L': [], 'loss': [], 'constr': []}
    #     for epoch in range(epochs):
    #         _epoch_best_loss = np.inf
    #         for lag_iter in range(maxiter):
    #             for i, data in enumerate(dataloader):
    #                 self._optimizer.zero_grad()
    #                 inputs, labels = data
    #                 outputs = self.net(inputs)
    #                 constraint_eval = self._constraint_fn(self.net)
    #                 loss_eval = self._loss_fn(outputs, labels)

    #                 if loss_eval < _epoch_best_loss:
    #                     _epoch_best_loss = loss_eval        
                    
    #                 L = loss_eval + self._lambda @ constraint_eval + 0.5*self._ss*torch.sum(torch.square(constraint_eval))
    #                 L.backward()
    #                 self._optimizer.step()

    #                 ###
    #                 if verbose:
    #                     print(f'{epoch}, {lag_iter}, {i}, {loss_eval.detach().item()}, {constraint_eval.detach().item()}', end='\r')
    #                 ###

    #                 self.history['L'].append(L)
    #                 self.history['loss'].append(loss_eval)
    #                 self.history['constr'].append(constraint_eval)
            
    #         if _epoch_best_loss < _total_best_loss:
    #             _total_best_loss = _epoch_best_loss
    #         else:
    #             _no_loss_improvement_epochs += 1

    #         with torch.no_grad():
    #             constr = self._constraint_fn(self.net)
    #             if constr < con_stopping_tol and _no_loss_improvement_epochs > early_stopping:
    #                 break
    #             if constr < (1/con_decrease_tol)*_prev_constr:
    #                 self._lambda += self._ss*constr
    #                 self._ss *= self._t
    #                 _prev_constr = constr
