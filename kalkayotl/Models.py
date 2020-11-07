'''
Copyright 2020 Javier Olivares Romero

This file is part of Kalkayotl.

	Kalkayotl is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Kalkayotl is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Kalkayotl.  If not, see <http://www.gnu.org/licenses/>.
'''
import sys
import numpy as np
import pymc3 as pm
from pymc3 import Model
import theano
from theano import tensor as tt, printing

from kalkayotl.Transformations import Iden,pc2mas,cartesianToSpherical,phaseSpaceToAstrometry,phaseSpaceToAstrometry_and_RV
from kalkayotl.Priors import EDSD,EFF,King,MvEFF,MvKing

################################## Model 1D ####################################
class Model1D(Model):
	'''
	Model to infer the distance of a series of stars
	'''
	def __init__(self,mu_data,tau_data,
		prior="Gaussian",
		parameters={"location":None,"scale": None},
		hyper_alpha=[100,10],
		hyper_beta=[10],
		hyper_gamma=None,
		hyper_delta=None,
		transformation="mas",
		parametrization="non-central",
		name='1D', model=None):
		super().__init__(name, model)

		#------------------- Data ------------------------------------------------------
		self.N = len(mu_data)

		if self.N == 0:
			sys.exit("Data has length zero!. You must provide at least one data point")

		#-------------------------------------------------------------------------------

		#============= Transformations ====================================

		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			Transformation = pc2mas

		else:
			sys.exit("Transformation is not accepted")

		if parametrization == "non-central":
			print("Using non central parametrization.")
		else:
			print("Using central parametrization.")
		#==================================================================

		#================ Hyper-parameters =====================================
		if hyper_delta is None:
			shape = ()
		else:
			shape = len(hyper_delta)

		#------------------------ Location ----------------------------------
		if parameters["location"] is None:
			pm.Normal("loc",mu=hyper_alpha[0],sigma=hyper_alpha[1],shape=shape)

		else:
			self.loc = parameters["location"]

		#------------------------ Scale ---------------------------------------
		if parameters["scale"] is None:
			pm.Gamma("scl",alpha=2.0,beta=2.0/hyper_beta,shape=shape)
		else:
			self.scl = parameters["scale"]
		#========================================================================

		#================= True values ========================================================
		#--------- Cluster oriented prior-----------------------------------------------
		if prior is "Uniform":
			if parametrization == "central":
				pm.Uniform("source",lower=self.loc-self.scl,upper=self.loc+self.scl,shape=self.N)
			else:
				pm.Uniform("offset",lower=-1.,upper=1.,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior is "Gaussian":
			if parametrization == "central":
				pm.Normal("source",mu=self.loc,sd=self.scl,shape=self.N)
			else:
				pm.Normal("offset",mu=0.0,sd=1.0,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior is "GMM":
			# break symmetry and avoids inf in advi
			pm.Potential('order_means', tt.switch(self.loc[1]-self.loc[0] < 0, -1e20, 0))

			if parameters["weights"] is None:
				pm.Dirichlet("weights",a=hyper_delta,shape=shape)	
			else:
				self.weights = parameters["weights"]

			if parametrization == "central":
				pm.NormalMixture("source",w=self.weights,
					mu=self.loc,
					sigma=self.scl,
					comp_shape=1,
					shape=self.N)
			else:
				pm.Normal("offset",mu=0.0,sd=1.0,shape=self.N)
				# latent cluster of each observation
				component = pm.Categorical("component",p=self.weights,shape=self.N)
				pm.Deterministic("source",self.loc[component] + self.scl[component]*self.offset) 

		elif prior is "EFF":
			if parameters["gamma"] is None:
				pm.Gamma("x",alpha=2.0,beta=2.0/hyper_gamma)
				pm.Deterministic("gamma",1.0+self.x)
			else:
				self.gamma = parameters["gamma"]

			if parametrization == "central":
				EFF("source",location=self.loc,scale=self.scl,gamma=self.gamma,shape=self.N)
			else:
				EFF("offset",location=0.0,scale=1.0,gamma=self.gamma,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)

		elif prior is "King":
			if parameters["rt"] is None:
				pm.Gamma("x",alpha=2.0,beta=2.0/hyper_gamma)
				pm.Deterministic("rt",1.0+self.x)
			else:
				self.rt = parameters["rt"]

			if parametrization == "central":
				King("source",location=self.loc,scale=self.scl,rt=self.rt,shape=self.N)
			else:
				King("offset",location=0.0,scale=1.0,rt=self.rt,shape=self.N)
				pm.Deterministic("source",self.loc + self.scl*self.offset)
			
		#---------- Galactic oriented prior ---------------------------------------------
		elif prior is "EDSD":
			EDSD("source",scale=self.scl,shape=self.N)
		
		else:
			sys.exit("The specified prior is not implemented")
		#-----------------------------------------------------------------------------
		#=======================================================================================
		# print_ = tt.printing.Print("source")(self.source)
		#----------------- Transformations ----------------------
		true = Transformation(self.source)

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=true, tau=tau_data,observed=mu_data)
		#------------------------------------------------------------------------------
####################################################################################################

############################ ND Model ###########################################################
class Model3D(Model):
	'''
	Model to infer the N-dimensional parameter vector of a cluster
	'''
	def __init__(self,dimension,mu_data,tau_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		hyper_eta=None,
		transformation=None,
		reference_system="ICRS",
		parametrization="non-central",
		name='', model=None):

		assert isinstance(dimension,int), "dimension must be integer!"
		assert dimension is 3,"Not a valid dimension!"

		super().__init__("3D", model)

		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/3)
		if N == 0:
			sys.exit("Data has length zero! You must provide at least one data point.")
		#-------------------------------------------------------------------------------

		print("Using {0} parametrization".format(parametrization))

		#============= Transformations ====================================
		assert transformation is "pc","3D model only works with 'pc' transformation"

		if reference_system is "ICRS":
				Transformation = cartesianToSpherical
		elif reference_system is "Galactic":
				Transformation = GalacticToSpherical
		else:
			sys.exit("Reference system not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		#----------------- Mixture prior families ----------------------------
		if prior in ["GMM","CGMM"]:
			#------------- Shapes -------------------------
			n_gauss = len(hyper_delta)

			loc  = theano.shared(np.zeros((n_gauss,3)))
			chol = theano.shared(np.zeros((n_gauss,3,3)))
			#----------------------------------------------

			#----------- Locations ------------------------------------------
			if parameters["location"] is None:
				if prior in ["CGMM"]:
					#----------------- Concentric prior --------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1]) for j in range(3) ]

					loci = pm.math.stack(location,axis=1)

					for i in range(shape):
						loc  = tt.set_subtensor(loc[i],loci)
					#---------------------------------------------------------

				else:
					#----------- Non-concentric prior ----------------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1],
								shape=n_gauss) for j in range(3) ]
					
					loc = pm.math.stack(location,axis=1)
					#---------------------------------------------------------
				#-------------------------------------------------------------------
			else:
				for i in range(n_gauss):
					loc  = tt.set_subtensor(loc[i],np.array(parameters["location"][i]))

			#---------- Covariance matrices -----------------------------------
			if parameters["scale"] is None:
				for i in range(n_gauss):
					choli, corri, stdsi = pm.LKJCholeskyCov("scl_{0}".format(i), 
										n=3, eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
										alpha=2.0,beta=1.0/hyper_beta),
										compute_corr=True)
				
					chol = tt.set_subtensor(chol[i],choli)

			else:
				sys.exit("Not yet implemented.")
			#--------------------------------------------------------------------
		#---------------------------------------------------------------------------------

		#-------------- Non-mixture prior families ----------------------------------
		else:
			#--------- Location ----------------------------------
			if parameters["location"] is None:
				location = [ pm.Normal("loc_{0}".format(i),
							mu=hyper_alpha[i][0],
							sigma=hyper_alpha[i][1]) for i in range(3) ]

				#--------- Join variables --------------
				loc = pm.math.stack(location,axis=1)

			else:
				loc = parameters["location"]
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				chol, corr, stds = pm.LKJCholeskyCov("scl", n=3, eta=hyper_eta, 
						sd_dist=pm.Gamma.dist(alpha=2.0,beta=1.0/hyper_beta),
						compute_corr=True)
			else:
				sys.exit("Not yet implemented.")
			#--------------------------------------------------------------
		#----------------------------------------------------------------------------
		#==============================================================================

		#===================== True values ============================================		
		if prior == "Gaussian":
			if parametrization == "central":
				pm.MvNormal("source",mu=loc,chol=chol,shape=(N,3))
			else:
				pm.Normal("offset",mu=0,sigma=1,shape=(N,3))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior == "King":
			if parameters["rt"] is None:
				pm.Gamma("x",alpha=2.0,beta=1.0/hyper_gamma)
				pm.Deterministic("rt",1.001+self.x)
			else:
				self.rt = parameters["rt"]

			if parametrization == "central":
				MvKing("source",location=loc,chol=chol,rt=self.rt,shape=(N,3))
			else:
				MvKing("offset",location=np.zeros(3),chol=np.eye(3),rt=self.rt,shape=(N,3))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior is "EFF":
			if parameters["gamma"] is None:
				pm.Gamma("x",alpha=2.0,beta=1.0/hyper_gamma)
				pm.Deterministic("gamma",3.001+self.x )
			else:
				self.gamma = parameters["gamma"]

			if parametrization == "central":
				MvEFF("source",location=loc,chol=chol,gamma=self.gamma,shape=(N,3))
			else:
				MvEFF("offset",location=np.zeros(3),chol=np.eye(3),gamma=self.gamma,shape=(N,3))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior in ["GMM","CGMM"]:
			pm.Dirichlet("weights",a=hyper_delta,shape=shape)

			comps = [ pm.MvNormal.dist(mu=loc[i],chol=chol[i]) for i in range(n_gauss)]

			#---- Sample from the mixture ----------------------------------
			pm.Mixture("source",w=self.weights,comp_dists=comps,shape=(N,3))
		
		else:
			sys.exit("The specified prior is not supported")
		#=================================================================================

		#----------------------- Transformation---------------------------------------
		transformed = Transformation(self.source)
		#-----------------------------------------------------------------------------

		#------------ Flatten --------------------------------------------------------
		true = pm.math.flatten(transformed)
		#----------------------------------------------------------------------------

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=true, tau=tau_data,observed=mu_data)
		#------------------------------------------------------------------------------

############################ 6D Model ###########################################################
class Model6D(Model):
	'''
	Model to infer the 6-dimensional parameter vector of a cluster
	'''
	def __init__(self,dimension,mu_data,tau_data,
		prior="Gaussian",
		parameters={"location":None,"scale":None},
		hyper_alpha=None,
		hyper_beta=None,
		hyper_gamma=None,
		hyper_delta=None,
		hyper_eta=None,
		transformation=None,
		parametrization="non-central",
		name='', model=None):

		assert isinstance(dimension,int), "dimension must be integer!"
		assert dimension is 6,"Not a valid dimension!"

		# 2) call super's init first, passing model and name
		# to it name will be prefix for all variables here if
		# no name specified for model there will be no prefix
		super().__init__("6D", model)
		# now you are in the context of instance,
		# `modelcontext` will return self you can define
		# variables in several ways note, that all variables
		# will get model's name prefix

		#------------------- Data ------------------------------------------------------
		N = int(len(mu_data)/6)
		if N == 0:
			sys.exit("Data has length zero! You must provide at least one data point.")
		#-------------------------------------------------------------------------------

		print("Using {0} parametrization".format(parametrization))

		#============= Transformations ====================================


		if transformation is "mas":
			Transformation = Iden

		elif transformation is "pc":
			if D is 3:
				Transformation = cartesianToSpherical
			elif D is 6:
				Transformation = phaseSpaceToAstrometry_and_RV
		else:
			sys.exit("Transformation is not accepted")
		#==================================================================

		#================ Hyper-parameters =====================================
		#----------------- Mixture prior families ----------------------------
		if prior in ["GMM","GUM","CGMM"]:
			#------------- Shapes -------------------------
			shape = len(hyper_delta)
			
			if prior is "GUM":
				n_gauss = shape -1
			else:
				n_gauss = shape

			loc  = theano.shared(np.zeros((n_gauss,D)))
			chol = theano.shared(np.zeros((n_gauss,D,D)))
			#----------------------------------------------

			#----------- Locations ------------------------------------------
			if parameters["location"] is None:
				if prior is "CGMM":
					#----------------- Concentric prior --------------------
					location = [ pm.Normal("loc_{0}".format(j),
								mu=hyper_alpha[j][0],
								sigma=hyper_alpha[j][1]) for j in range(D) ]

					loci = pm.math.stack(location,axis=1)

					for i in range(n_gauss):
						loc  = tt.set_subtensor(loc[i],loci)
					#---------------------------------------------------------

				else:
					#----------- Non-concentric prior ----------------------------
					for i in range(n_gauss):
						location = [ pm.Normal("loc_{0}_{1}".format(i,j),
									mu=hyper_alpha[j][0],
									sigma=hyper_alpha[j][1]) for j in range(D) ]
						
						loci = pm.math.stack(location,axis=1)

						loc  = tt.set_subtensor(loc[i],loci)
					#---------------------------------------------------------
					
					if prior is "GUM":
						#---------- Gaussian+Uniform ------------------------------
						location_unif = [ pm.Normal("loc_unif_{0}".format(i),
									mu=hyper_alpha[i][0],
									sigma=hyper_alpha[i][1]) for i in range(D) ]

						loc_unif = pm.math.stack(location_unif,axis=1)
						#----------------------------------------------------------
				#-------------------------------------------------------------------
			else:
				for i in range(n_gauss):
					loc  = tt.set_subtensor(loc[i],np.array(parameters["location"][i]))

				if prior is "GUM":
					loc_unif = pm.math.stack(np.array(parameters["location"][-1]),axis=1)

			#---------- Covariance matrices -----------------------------------
			if parameters["scale"] is None:
				for i in range(n_gauss):
					choli, corri, stdsi = pm.LKJCholeskyCov("scl_{0}".format(i), 
										n=D, eta=hyper_eta, 
										sd_dist=pm.Gamma.dist(
										alpha=2.0,beta=2.0/hyper_beta),
										compute_corr=True)
				
					chol = tt.set_subtensor(chol[i],choli)

				if prior == "GUM":
					scl_unif = pm.Gamma("scl_unif",alpha=2.0,beta=2.0/hyper_beta,shape=D)
			else:
				sys.exit("Not yet implemented.")
			#--------------------------------------------------------------------
		#---------------------------------------------------------------------------------

		#-------------- Non-mixture prior families ----------------------------------
		else:
			#--------- Location ----------------------------------
			if parameters["location"] is None:
				location = [ pm.Normal("loc_{0}".format(i),
							mu=hyper_alpha[i][0],
							sigma=hyper_alpha[i][1]) for i in range(D) ]

				#--------- Join variables --------------
				loc = pm.math.stack(location,axis=1)

			else:
				loc = parameters["location"]
			#------------------------------------------------------

			#---------- Covariance matrix ------------------------------------
			if parameters["scale"] is None:
				chol, corr, stds = pm.LKJCholeskyCov("scl", n=D, eta=hyper_eta, 
						sd_dist=pm.Gamma.dist(alpha=2.0,beta=2.0/hyper_beta),
						compute_corr=True)
			else:
				sys.exit("Not yet implemented.")
			#--------------------------------------------------------------
		#----------------------------------------------------------------------------
		#==============================================================================

		#===================== True values ============================================		
		if prior == "Gaussian":
			if parametrization == "central":
				pm.MvNormal("source",mu=loc,chol=chol,shape=(N,D))
			else:
				pm.Normal("offset",mu=0,sigma=1,shape=(N,D))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior == "King":
			if parameters["rt"] is None:
				pm.Gamma("x",alpha=2.0,beta=2.0/hyper_gamma)
				pm.Deterministic("rt",1.0+self.x)
			else:
				self.rt = parameters["rt"]

			if parametrization == "central":
				MvKing("source",location=loc,scale=scl,rt=self.rt,shape=(N,D))
			else:
				King("offset",location=0.0,scale=1.0,rt=self.rt,shape=(N,D))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior is "EFF":
			if parameters["gamma"] is None:
				pm.Gamma("x",alpha=2.0,beta=2.0/hyper_gamma)
				pm.Deterministic("gamma",1.0+self.x)
			else:
				self.gamma = parameters["gamma"]

			if parametrization == "central":
				MvEFF("source",location=loc,scale=scl,gamma=self.gamma,shape=(N,D))
			else:
				EFF("offset",location=0.0,scale=1.0,gamma=self.gamma,shape=(N,D))
				pm.Deterministic("source",loc + tt.nlinalg.matrix_dot(self.offset,chol))

		elif prior in ["GMM","CGMM","GUM"]:
			pm.Dirichlet("weights",a=hyper_delta,shape=shape)

			comps = [ pm.MvNormal.dist(mu=loc[i],chol=chol[i]) for i in range(n_gauss)]

			if prior is "GUM":
				comps.extend(MvUniform.dist(location=loc_unif,scale=scl_unif))

			#---- Sample from the mixture ----------------------------------
			pm.Mixture("source",w=self.weights,comp_dists=comps,shape=(N,D))
		
		else:
			sys.exit("The specified prior is not supported")
		#=================================================================================

		#----------------------- Transformation---------------------------------------
		transformed = Transformation(self.source)
		#-----------------------------------------------------------------------------

		#------------ Flatten --------------------------------------------------------
		true = pm.math.flatten(transformed)
		#----------------------------------------------------------------------------

		#----------------------- Likelihood ----------------------------------------
		pm.MvNormal('obs', mu=true, tau=tau_data,observed=mu_data)
		#------------------------------------------------------------------------------