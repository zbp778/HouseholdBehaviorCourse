import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.rho = 0.98 # discount factor

        par.beta_0 = 0.1 # weight on labor dis-utility (constant)
        par.beta_1 = 0.03 # additional weight on labor dis-utility (children)
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax

        # children
        par.p_birth = 0.1

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 50 #70 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in human capital grid
        par.Nk = 20 #30 # number of grid points in human capital grid    

        par.Nn = 2 # number of children

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals

        # structural estimation
        par.min_time = -8
        par.max_time = 8

        # income process
        par.y_cons=0.1
        par.y_growth=0.01

        #childcare subsidy changes
        par.theta=0.05

        #par spouse
        par.p_spouse=0.8

        par.Ns=2 # number of spouses

    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # c. number of children and spouses grid
        par.n_grid = np.arange(par.Nn)
        par.s_grid = np.arange(par.Ns)

        # d. solution arrays
        shape = (par.T,par.Ns,par.Nn,par.Na,par.Nk)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)

        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)

        # h. vector of wages. Used for simulating elasticities
        par.w_vec = par.w * np.ones(par.T)

        # i. structural estimation
        par.birth = np.zeros(sim.n.shape,dtype=np.int_)
        par.periods = np.tile([t for t in range(par.simT)],(par.simN,1))
        par.event_grid = np.arange(par.min_time,par.max_time+1)
        par.event_hours = np.nan + np.zeros(par.event_grid.size)

        # j. parameters to be estimated
        sim.beta_1N=8
        sim.beta_1_min=0.000001
        sim.beta_1_max=0.1
        sim.distance_grid= np.nan + np.zeros(sim.beta_1N)
        sim.beta_1_grid=np.linspace(sim.beta_1_min,sim.beta_1_max,sim.beta_1N)
        sim.beta_1_grid=np.round(sim.beta_1_grid,2) #rounding of beta guesses
        
        # k. income process
        par.y_vec = par.y_cons * (1+par.y_growth)**np.arange(par.T)

        #l. draws used to simulate spouse appearance
        np.random.seed(9211)
        sim.draws_uniform_spouse = np.random.uniform(size=shape)
        sim.s=np.zeros(shape)
        cond=(sim.draws_uniform_spouse<=0.8)==1
        sim.s[cond]=1
        sim.s=sim.s.astype(int) #such that I can index in using sim.s

    ############
    # Solution #
    def solve(self,beta_1=0.03):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # a2. update beta_1
        par.beta_1 = beta_1
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            # i. loop over state variables: number of spouses, children, human capital and wealth in beginning of period
            for i_s,spouse in enumerate(par.s_grid):
                for i_n,kids in enumerate(par.n_grid):
                    for i_a,assets in enumerate(par.a_grid):
                        for i_k,capital in enumerate(par.k_grid):
                            idx = (t,i_s,i_n,i_a,i_k)

                            # ii. find optimal consumption and hours at this level of wealth in this period t.

                            if t==par.T-1: # last period
                                obj = lambda x: self.obj_last(x[0],assets,capital,kids,spouse)

                                constr = lambda x: self.cons_last(x[0],assets,capital,kids,spouse)
                                nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf,keep_feasible=True)

                                # call optimizer
                                hours_min = (- assets -par.y_vec[t]*spouse +par.theta*kids) / self.wage_func(capital,t) + 1.0e-5 # minimum amout of hours that ensures positive consumption
                                hours_min = np.maximum(hours_min,2.0)
                                init_h = np.array([hours_min]) if i_a==0 else np.array([sol.h[t,i_s,i_n,i_a-1,i_k]]) # initial guess on optimal hours

                                res = minimize(obj,init_h,bounds=((0.0,np.inf),),constraints=nlc,method='trust-constr')

                                # store results
                                sol.c[idx] = self.cons_last(res.x[0],assets,capital,kids,spouse)
                                sol.h[idx] = res.x[0]
                                sol.V[idx] = -res.fun

                            else:
                                
                                # objective function: negative since we minimize
                                obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,kids,spouse,t)  

                                # bounds on consumption 
                                lb_c = 0.000001 # avoid dividing with zero
                                ub_c = np.inf

                                # bounds on hours
                                lb_h = 0.0
                                ub_h = np.inf 

                                bounds = ((lb_c,ub_c),(lb_h,ub_h))
                    
                                # call optimizer
                                init = np.array([lb_c,1.0]) if (i_n == 0 & i_a==0 & i_k==0 & i_s==0) else res.x  # initial guess on optimal consumption and hours
                                res = minimize(obj,init,bounds=bounds,method='L-BFGS-B') 
                            
                                # store results
                                sol.c[idx] = res.x[0]
                                sol.h[idx] = res.x[1]
                                sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital,kids,spouse):
        par = self.par
        income = self.wage_func(capital,par.T-1) * hours + par.y_vec[par.T-1]*spouse -par.theta*kids
        cons = assets + income
        return cons

    def obj_last(self,hours,assets,capital,kids,spouse):
        cons = self.cons_last(hours,assets,capital,kids,spouse)
        return - self.util(cons,hours,kids)    

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,kids,spouse,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = self.util(cons,hours,kids)
        
        # d. *expected* continuation value from savings
        income = self.wage_func(capital,t) * hours + par.y_vec[t]*spouse -par.theta*kids
        a_next = (1.0+par.r)*(assets + income - cons)
        k_next = capital + hours

        # no birth, no spouse next
        kids_next = kids
        V_next = sol.V[t+1,0,kids_next] #Set spouse to 0 in next period.
        V_next_no_b_no_s = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        #no birth, spouse next
        V_next = sol.V[t+1,1,kids_next] #Set spouse to 1 in next period.
        V_next_no_b_s = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        # birth, no spouse next
        if (kids>=(par.Nn-1)): # if kids is at max already
            # cannot have more children
            V_next_b_no_s = V_next_no_b_no_s
        else:
            kids_next = kids + 1
            V_next = sol.V[t+1,0,kids_next] #Set spouse to 1 in next period.
            V_next_b_no_s = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        # birth, spouse next
        if (kids>=(par.Nn-1)): # if kids is at max already
            # cannot have more children
            V_next_b_s = V_next_no_b_s
        else:
            kids_next = kids + 1
            V_next = sol.V[t+1,1,kids_next] #Set spouse to 1 in next period.
            V_next_b_s = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        if spouse==0: #if no spouse in current period
            EV_next = (par.p_spouse)*V_next_no_b_s + (1-par.p_spouse)*V_next_no_b_no_s
        else: #if spouse in current period
            EV_next = par.p_birth*par.p_spouse * V_next_b_s + par.p_birth*(1-par.p_spouse)*V_next_b_no_s + (1-par.p_birth)*(par.p_spouse)*V_next_no_b_s + (1-par.p_birth)*(1-par.p_spouse)*V_next_no_b_no_s
        
        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours,kids):
        par = self.par

        beta = par.beta_0 + par.beta_1*kids

        return (c)**(1.0+par.eta) / (1.0+par.eta) - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital,t):
        # after tax wage rate
        par = self.par

        return (1.0 - par.tau )* par.w_vec[t] * (1.0 + par.alpha * capital)

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.n[i,0] = sim.n_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.s[i,t],sim.n[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t] + par.y_vec[t]*sim.s[i,t] -par.theta*sim.n[i,t]
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    birth = 0 
                    #conditional on drawing drawing a kids=1, not having a kid already and having a spouse in current period.
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1)) & (sim.s[i,t]==1)):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth
                    

    ##############
    # Evaluate Child Penalty#
    def evaluate_diff_cp(self,beta_1):

        """ Evaluate average drop in hours at t=0 relative to t=-1"""
    
        # a. unpack
        par = self.par
        sim = self.sim
        sol = self.sol

        # a2. update parameters
        par.beta_1 = beta_1
        #print(beta_1)
        # b. solve model using new parameters
        self.solve(beta_1)

        # b. simulate model using new parameters
        self.simulate()

        # c. define birth vector
        par.birth[:,1:] = (sim.n[:,1:] - sim.n[:,:-1]) > 0
    
        #d. define time since birth
        time_of_birth = np.max(par.periods * par.birth, axis=1)
        I = time_of_birth>0
        time_of_birth[~I] = -1000 # never has a child
        time_of_birth = np.transpose(np.tile(time_of_birth , (par.simT,1)))
        time_since_birth = par.periods - time_of_birth

        #e. relative drop in hours after birth
        for t,time in enumerate(par.event_grid):
            #Average over the hours t years before birth by finding the persons'
            #working hours who had given birth t years before.
            par.event_hours[t] = np.mean(sim.h[time_since_birth==time])

        # relative to period before birth
        event_hours_rel = par.event_hours - par.event_hours[par.event_grid==-1]

        #Deviation
        return (-0.1-event_hours_rel[-par.min_time])**2
    
    ##############
    #Structural Estimation#
    def structural_estimation(self):

        """" estimate parameters """
        #a. par, sol
        par = self.par
        sol = self.sol
        sim = self.sim

        #b. callable objective function
        obj = lambda x: self.evaluate_diff_cp(x)

        #a. grid search
        for ib,beta_1 in enumerate(sim.beta_1_grid):
            #b. callable objective function
            #print(f"grid_count={ib}")
            sim.distance_grid[ib] = obj(beta_1)

        min_distance_index=np.argmin(sim.distance_grid)
        min_distance_value=np.sqrt(sim.distance_grid[min_distance_index])
        beta_1_struc=sim.beta_1_grid[min_distance_index]

        return (sim.beta_1_grid,sim.distance_grid, beta_1_struc)


