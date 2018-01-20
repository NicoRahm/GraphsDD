import numpy as np 
import scipy.stats as stats
from igraph import *
import time
import copy
from scipy.spatial.distance import hamming
import editdistance
import matplotlib.pyplot as plt 

def truncated_power_law(a, N):
	# Generates N samples from the truncated power law p(x) = Z/x^a 
	# (we truncate at the number of vertices minus one to ensure the 
	# DD is suitable)
	m = N - 1
	x = np.arange(1, m+1, dtype='float')
	pmf = 1/x**a
	pmf /= pmf.sum()
	return(stats.rv_discrete(values=(range(1, m+1), pmf)).rvs(size = N))

def normal_order(degrees): 
	# Order vertices in normal order according to the prescribed degrees
	idx = np.argsort(degrees, order = ('in', 'out'))
	return(idx[::-1])

def construct_simple_graph_HH(degrees, verbose = False): 
	# Construct a directed with the exact prescribed degree sequence 
	# using Havel-Hakimi greedy algorithm (in the cases it is possible)
	g = Graph(directed = True)
	n_Vertices = degrees.shape[0]
	g.add_vertices(n_Vertices)

	all_edges = compute_edges_HH(degrees, verbose = verbose)

	g.add_edges(all_edges)

	return(g)

def compute_edges_HH(degrees, verbose = False): 
	# Construct a directed with the exact prescribed degree sequence 
	# using Havel-Hakimi greedy algorithm (in the cases it is possible)
	t = 0
	n_Vertices = degrees.shape[0]

	# Verify that the sum of in-degrees is equal to the sum of out-degrees
	sum_in = 0
	sum_out = 0
	for i in range(n_Vertices): 
		sum_out += degrees[i][1]
		sum_in += degrees[i][0]

	if sum_out != sum_in: 
		raise RuntimeError("in and out degrees sum do not match...")

	# Initialize the ordering of the vertices 
	idx = np.array([i for i in range(n_Vertices)])
	current_degrees = copy.deepcopy(degrees)
	idx = normal_order(current_degrees)

	all_edges = []

	# first_non_zero = 1

	# Loop over the ordered vertices 
	start1 = time.time()
	t_sort = 0
	t_reorder = 0
	n = np.count_nonzero(np.array(current_degrees.tolist())[:,1])
	while  n > 0:		
		work_node = 0
		work_node_idx = 0
		for i in idx:				 
			if current_degrees[i][1] > 0:
				work_node = i
				break
			else: 
				work_node_idx += 1
		d_plus = current_degrees[work_node][1] # out degree of considered vertex

		# Construct the out-neighbourhood 
		k = 0
		edge_added = 0
		in_nodes_idx = []
		while edge_added < d_plus: 
			# Look for the next vertex which still have a non-empty in-neighbourhood
			if k >= n_Vertices:
				print("ERROR : No graph with this DD...")
				return([])
			if verbose:
				print((work_node, idx[k]))

			if idx[k] != work_node and current_degrees[idx[k]][0] > 0:
				# Add the corresponding edge to the edge set
				all_edges.append(((work_node, idx[k])))

				# Update the in-degree 
				current_degrees[idx[k]][0] -= 1
				in_nodes_idx.append(k)
				k+=1
				edge_added += 1

			else: 
				k+=1

		# Update the out-degree
		current_degrees[work_node][1] = 0
		
		# start = time.time()
		# idx_reorder = reorder(current_degrees, idx, work_node_idx, in_nodes_idx, verbose = False)
		# t_reorder += time.time() - start
		
		start = time.time()
		idx = normal_order(current_degrees)
		t_sort += time.time() - start 
		
		n-=1
	t = time.time() - start1
	# print("Total execution time : ", t, "s")
	# print("Mean sort time :", t_sort, "s")
	# print("Mean reorder time :", t_reorder, "s")


	return(all_edges)

def reorder(degrees, idx, work_node_idx, in_nodes_idx, verbose = False): 

	if verbose:
		print("in_nodes_idx :", in_nodes_idx)
	for i in range(len(in_nodes_idx)): 
		if work_node_idx < in_nodes_idx[i]:
			in_nodes_idx[i] -= 1
	if verbose:
		print("in_nodes_idx :", in_nodes_idx)

	work_node = idx[work_node_idx]
	if verbose:
		print("Work node :", work_node)
	new_idx = idx.tolist()
	new_idx = new_idx[:work_node_idx] + new_idx[work_node_idx+1:]

	index = in_nodes_idx[-1]
	if verbose:
		print('Index :', index)
		print("New_idx : ", new_idx)
		print(degrees[new_idx])

	if index < len(new_idx) - 1:
		while index >= in_nodes_idx[0] and (degrees[new_idx[index]][0] < degrees[new_idx[index+1]][0] or (degrees[new_idx[index]][0] == degrees[new_idx[index+1]][0] and degrees[new_idx[index]][1] < degrees[new_idx[index+1]][1])):  
			d_to_swap = 1
			while index + d_to_swap + 1 < len(new_idx) and (degrees[new_idx[index]][0] < degrees[new_idx[index+d_to_swap+1]][0] or (degrees[new_idx[index]][0] == degrees[new_idx[index+d_to_swap+1]][0] and degrees[new_idx[index]][1] < degrees[new_idx[index+d_to_swap+1]][1])):
				d_to_swap += 1
			for k in range(d_to_swap):
				mem = new_idx[index + k]
				new_idx[index + k] = new_idx[index+k+1]
				new_idx[index + k+1] = mem
			index -= 1
			if verbose: 
				print("New_idx :", new_idx)

	insert_idx = 0
	while insert_idx < len(new_idx) and (degrees[work_node][0] < degrees[new_idx[insert_idx]][0] or (degrees[work_node][0] == degrees[new_idx[insert_idx]][0] and degrees[work_node][1] < degrees[new_idx[insert_idx]][1])): 
		insert_idx += 1
	if verbose:
		print("Insert idx :", insert_idx)
	new_idx = new_idx[:insert_idx] + [work_node] + new_idx[insert_idx:]
	return(np.array(new_idx))


def generate_degree_sequence(n_Vertices, in_distribution = "poisson", out_distribution = "poisson", l = 5, a = 3):
	# Generates a degree sequence which follows a given distribution

	# Sample in-degrees from distribution
	if in_distribution == 'poisson':
		in_degrees = np.random.poisson(lam = l, size = n_Vertices)  
	elif in_distribution == 'power':
		in_degrees = truncated_power_law(a = a , N = n_Vertices)  
	else: 
		raise RuntimeError("Unknown distribution... Please chose a degree distribution from this list : poisson, power")  

	# Sample out-degrees from distribution
	if out_distribution == 'poisson':
		out_degrees = np.random.poisson(lam = l, size  = n_Vertices) + 1
	elif out_distribution == 'power':
		out_degrees = truncated_power_law(a = a, N = n_Vertices)
	else: 
		raise RuntimeError("Unknown distribution... Please chose a degree distribution from this list : poisson, power")  

	# Make the degree sequence suitable for a graph (sum of in-degrees equal to sum of out-degrees)
	diff = np.sum(in_degrees) - np.sum(out_degrees)
	while diff != 0:
		u = np.random.uniform(0,1,size = 1)
		i = np.random.randint(0, n_Vertices, 1)
		if diff > 0: 
			if u > 0.5: 
				if in_degrees[i] > 0:
					in_degrees[i] -= 1
					diff -= 1
			else: 
				out_degrees[i] += 1
				diff -= 1
		else: 
			if u > 0.5:
				in_degrees[i] += 1
				diff += 1
			else: 
				if out_degrees[i] > 0:
					out_degrees[i] -= 1
					diff += 1



	# Construct the data structure to stor the degree sequence
	degrees = np.array([(0,0) for k in range(n_Vertices)],  dtype = [('in', '<i4'), ('out', '<i4')])

	for i in range(n_Vertices): 
		if degrees[i][1] == n_Vertices:
			degrees[i][1] -= 1
		if degrees[i][0] == n_Vertices:
			degrees[i][0] -= 1
		degrees[i][1] = out_degrees[i]
		degrees[i][0] = in_degrees[i]

	return(degrees)

def swap_2_edges(g, edge1, edge2): 
	new_edge1 = (edge1.tuple[0], edge2.tuple[1])
	new_edge2 = (edge2.tuple[0], edge1.tuple[1])
	g.delete_edges([edge1, edge2])

	g.add_edges([new_edge1, new_edge2])

def swap_k_edges(g, edges): 
	edges = list(edges)
	
	l = len(edges)
	for i in range(1, l+1): 
		new_edge1 = (edges[i-1].tuple[0], edges[i%l].tuple[1])
		new_edge2 = (edges[i%l].tuple[0], edges[i-1].tuple[1])

		g.add_edges([new_edge1, new_edge2])
	
	g.delete_edges(edges)

def swap_random_edges(g, n_swap): 

	for i in range(n_swap): 
		edges = []
		list_edges = []
		for e in g.es: 
			edges.append(e)
			list_edges.append(e.tuple)

		edges = np.array(edges)
		edge1 = (0,0)
		edge2 = (0,0)
		while edge1[0] == edge2[1] or edge1[1] == edge2[0] or list_edges.count((edge1[0], edge2[1])) > 0 or list_edges.count((edge2[0], edge1[1])):
			selected_edges = np.random.choice(edges, 2, replace = False)
			edge1 = selected_edges[0].tuple
			edge2 = selected_edges[1].tuple
		swap_2_edges(g, selected_edges[0], selected_edges[1])

def swap_k_random_edges(g, n_swap, k = 4): 
	for i in range(n_swap): 
		edges = []
		list_edges = []
		for e in g.es: 
			edges.append(e)
			list_edges.append(e.tuple)



		edges = np.array(edges)

		self_loop = True
		multiple = True
		nb_it = 0
		while self_loop or multiple:
			selected_edges = np.random.choice(edges, k, replace = False)
			g_copy = copy.deepcopy(g)
			swap_k_edges(g_copy, selected_edges)
			self_loop = g_copy.is_loop().count(True) > 0
			multiple = g_copy.is_multiple().count(True) > 0
			if nb_it > 10000:
				print("Stopped at", nb_it - 1, "iterations for n_swap =", i + 1)
				break
			nb_it += 1
		if nb_it <= 10000:
			swap_k_edges(g, selected_edges)

def swap_seperates(g):
	n=g.vcount()
	ind1=np.random.randint(n)
	ld=g.get_adjlist("ALL")[ind1]
	if (len(ld)<2):
		return 0
	ind2=np.random.choice(ld)
	ld.remove(ind2)
	li=g.get_adjlist("ALL")[ind2]
	li.remove(ind1)
	if (len(li)==0):
		return 0
	ind3=np.random.choice(li)
	if (ind3 in ld):
		return 0
	l3=g.get_adjlist("OUT")[ind3]
	if (ind1 in l3):
		l3.remove(ind1)	
	if (ind2 in l3):
		l3.remove(ind2)
	if (len(l3)==0):
		return 0
	l4=g.get_adjlist("IN")[ind1]
	if (ind2 in l4):
		l4.remove(ind2)	
	if (ind3 in l4):
		l4.remove(ind3)
	if (len(l4)==0):
		return 0
	to_remove_ind1=np.random.choice(l4)
	to_remove_ind3=np.random.choice(l3)
	
	g.delete_edges((to_remove_ind1,ind1))
	g.delete_edges((ind3,to_remove_ind3))
	g.add_edges([(ind3,ind1), (to_remove_ind1,to_remove_ind3)])
	if(g.is_loop().count(True)>0 or  g1.is_multiple().count(True)   >0 ):
		g.delete_edges((ind3,ind1))
		g.delete_edges((to_remove_ind1,to_remove_ind3))
		g.add_edges([(to_remove_ind1,ind1), (ind3,to_remove_ind3)])		
		return 0	  
	return 1


def generate_erdos_renyi(n, p): 

	g = Graph(directed = True)
	g.add_vertices(n)

	for i in range(n):
		for j in range(n): 
			u = np.random.uniform(0,1)
			if j!= i and u < p:
				g.add_edges([(i,j)])

	return(g) 

def generate_watts_graph(n, K, beta): 
	# Generate a small-world graph following the Watts-Strogatz model: https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model

	g = Graph(directed = True)
	g.add_vertices(n)

	for i in range(n):
		for j in range(n):
			c = abs(i-j)%(n-1-K/2) 
			if c > 0 and c <= K/2:
				g.add_edges([(i,j)])

	edges = []
	list_edges = []
	for e in g.es: 
		edges.append(e)
		list_edges.append(e.tuple)

	for edge in list_edges: 
		if edge[0] < edge[1]: 
			u = np.random.uniform(0,1)
			if u < beta:
				k = np.random.randint(0,n)
				if (edge[0], k) not in list_edges and k != edge[0]: 
					g.delete_edges(edge)
					g.add_edges([(edge[0], k)])

	return(g)

def hist_degrees(g): 

	out_degrees =  g.degree(type = 'out')
	in_degrees =  g.degree(type = 'in')
	m = max(max(in_degrees), max(out_degrees))

	p, ax = plt.subplots(2)
	bins = [i - 0.5 for i in range(m+1)]
	ax[0].hist(in_degrees, bins = bins)
	ax[1].hist(out_degrees, bins = bins)

	plt.show()



########################

#######################
#### begin		  ####
#######################


########################
### create random edges and update the degree sequence, before applying the Havel-Hakimi greedy algorithm 
def construct_simple_graph_random_edge(degrees, verbose = False,n_rand=0):
	# n_rand = number of randomly created edges
	# Construct a directed with the exact prescribed degree sequence 
	# using Havel-Hakimi greedy algorithm (in the cases it is possible)

	g = Graph(directed = True)
	n_Vertices = degrees.shape[0]
	g.add_vertices(n_Vertices)

	# Verify that the sum of in-degrees is equal to the sum of out-degrees
	sum_in = 0
	sum_out = 0
	for i in range(n_Vertices): 
		sum_out += degrees[i][0]
		sum_in += degrees[i][1]

	if sum_out != sum_in: 
		raise RuntimeError("in and out degrees sum do not match...")
	# create random edges first
	rand_edges = []
	it=0	
	while(it < n_rand):
		l1=np.random.randint(degrees.shape[0])
		l2=np.random.randint(degrees.shape[0])
		if(degrees[l1][0] >0 and degrees[l2][1] >0):
			rand_edges.append(((l1, l2)))
			it+=1
			degrees[l1][0]=degrees[l1][0]-1						
			degrees[l2][1]=degrees[l2][1]-1
	
	g.add_edges(rand_edges)	
	if (g.is_loop().count(True) > 0 or g.is_multiple().count(True)  >0):
		raise RuntimeError("Initialization error, ... try again") 

	all_edges = compute_edges_HH(degrees, verbose = verbose)

	g.add_edges(all_edges)

	if (g.is_loop().count(True) > 0 or g.is_multiple().count(True)  >0):
		raise RuntimeError("Initialization error, ... try again") 

	return(g)
#######################
### create loop structure and update the degree sequence, before applying the Havel-Hakimi greedy algorithm 
def construct_simple_graph_loop(degrees, verbose = False,n_rand=0):
	# n_rand = length of the loop
	# Construct a directed with the exact prescribed degree sequence 
	# using Havel-Hakimi greedy algorithm (in the cases it is possible)

	g = Graph(directed = True)
	n_Vertices = degrees.shape[0]
	g.add_vertices(n_Vertices)

	# Verify that the sum of in-degrees is equal to the sum of out-degrees
	sum_in = 0
	sum_out = 0
	for i in range(n_Vertices): 
		sum_out += degrees[i][0]
		sum_in += degrees[i][1]

	if sum_out != sum_in: 
		raise RuntimeError("in and out degrees sum do not match...")
	# create random edges first
	rand_edges = []
	it=0	
	while(it < n_rand):
		if(it==0):		
			l1=np.random.randint(degrees.shape[0])
			l0=l1			
		else:
			l1=l2
		if(it==(n_rand-1)):		
			l2=l0
		else:
			l2=np.random.randint(degrees.shape[0])			
		if(degrees[l1][0] >0 and degrees[l2][1] >0):
			rand_edges.append(((l1, l2)))
			it+=1
			degrees[l1][0]=degrees[l1][0]-1						
			degrees[l2][1]=degrees[l2][1]-1
	
	g.add_edges(rand_edges)		
	if (g.is_loop().count(True) > 0 or g.is_multiple().count(True) >0 ):
		raise RuntimeError("Initialization error, ... try again")
	
	all_edges = compute_edges_HH(degrees, verbose = verbose)

	g.add_edges(all_edges)

	if (g.is_loop().count(True) > 0 or g.is_multiple().count(True) >0 ):
		raise RuntimeError("Initialization error, ... try again")

	return(g)
#######################
### create chain structure and update the degree sequence, before applying the Havel-Hakimi greedy algorithm 
def construct_simple_graph_chain(degrees, verbose = False,n_rand=0):
	# n_rand = length of the chain
	# Construct a directed with the exact prescribed degree sequence 
	# using Havel-Hakimi greedy algorithm (in the cases it is possible)

	g = Graph(directed = True)
	n_Vertices = degrees.shape[0]
	g.add_vertices(n_Vertices)

	# Verify that the sum of in-degrees is equal to the sum of out-degrees
	sum_in = 0
	sum_out = 0
	for i in range(n_Vertices): 
		sum_out += degrees[i][0]
		sum_in += degrees[i][1]

	if sum_out != sum_in: 
		raise RuntimeError("in and out degrees sum do not match...")
	# create random edges first
	rand_edges = []
	it=0	
	while(it < n_rand):
		if(it==0):		
			l1=np.random.randint(degrees.shape[0])	   
		else:
			l1=l2
		l2=np.random.randint(degrees.shape[0])			
		if(degrees[l1][0] >0 and degrees[l2][1] >0):
			rand_edges.append(((l1, l2)))
			it+=1
			degrees[l1][0]=degrees[l1][0]-1						
			degrees[l2][1]=degrees[l2][1]-1
	
	g.add_edges(rand_edges)		
	if (g.is_loop().count(True) > 0 or g.is_multiple().count(True) >0 ):
		raise RuntimeError("Initialization error, ... try again")
	
	all_edges = compute_edges_HH(degrees, verbose = verbose)

	g.add_edges(all_edges)

	if (g.is_loop().count(True) > 0 or g.is_multiple().count(True) >0 ):
		raise RuntimeError("Initialization error, ... try again")

	return(g)
#######################

# configuration model with repeat. Rejectin => repeat => not efficient.
def configuration_repeat(degrees):
	b=True
	degrees_array=np.asarray(degrees.tolist())
	while(b):
		g = Graph(directed = True)
		n_Vertices = degrees.shape[0]
		g.add_vertices(n_Vertices)
		edges = []
		current_degrees=degrees_array.copy()
		b2=True	
		while(b2):	
			out_list=	np.where(current_degrees[:,0]>0)[0]	
			in_list=	np.where(current_degrees[:,1]>0)[0]
			if(out_list.shape[0]==0 or in_list.shape[0]==0):
				b2=False  
				break									
			l1=np.random.choice( out_list )	 
			l2=np.random.choice( in_list ) 		
			edges.append(((l1, l2)))
			current_degrees[l1][0]=current_degrees[l1][0]-1						
			current_degrees[l2][1]=current_degrees[l2][1]-1
	
		g.add_edges(edges)		
		b=(g.is_loop().count(True) > 0 or g.is_multiple().count(True) >0 )
		#print(b)	
	return(g)
#######################
# configuration model with erase : approximate DD and not exact.
def configuration_erase(degrees):
	degrees_array=np.asarray(degrees.tolist())
	g = Graph(directed = True)
	n_Vertices = degrees.shape[0]
	g.add_vertices(n_Vertices)
	edges = []
	current_degrees=degrees_array.copy()
	b2=True	
	while(b2):	
		out_list=	np.where(current_degrees[:,0]>0)[0]	
		in_list=	np.where(current_degrees[:,1]>0)[0]
		if(out_list.shape[0]==0 or in_list.shape[0]==0):
			b2=False  
			break									
		l1=np.random.choice( out_list )	 
		l2=np.random.choice( in_list ) 		
		edges.append(((l1, l2)))
		current_degrees[l1][0]=current_degrees[l1][0]-1						
		current_degrees[l2][1]=current_degrees[l2][1]-1
	
	g.add_edges(edges)		
	print("Loop to erase?", g.is_loop().count(True))	
	print("Multiple to erase?", g.is_multiple().count(True))
	g.simplify()
	return(g)
	
#######################

# construct random graph without rejection with exact DD
def construct_simple_graph_sampling(degrees, verbose = False): 
	# Construct a directed with the exact prescribed degree sequence 
	# using Havel-Hakimi greedy algorithm (in the cases it is possible)

	g = Graph(directed = True)
	n_Vertices = degrees.shape[0]
	g.add_vertices(n_Vertices)

	# Verify that the sum of in-degrees is equal to the sum of out-degrees
	sum_in = 0
	sum_out = 0
	for i in range(n_Vertices): 
		sum_out += degrees[i][0]
		sum_in += degrees[i][1]

	if sum_out != sum_in: 
		raise RuntimeError("in and out degrees sum donot match...")

	# Initialize the ordering of the vertices 

	current_degrees = copy.deepcopy(degrees)

	all_edges = []
	
	#1) finding work node
	
	idx = np.array([i for i in range(n_Vertices)])
	idx = normal_order(current_degrees)
	
	work_node = idx[0]
		
	while np.count_nonzero(np.array(current_degrees.tolist())) > 0:
		work_idx=0
		work_node=idx[work_idx]
		

		while(current_degrees[work_node][1]==0):
			work_idx+=1	
			work_node=idx[work_idx]
		#2 initialize forbiddens
		forbiddens=[]	
		forbiddens.append(work_node)
		degrees_array=np.asarray(current_degrees.tolist())	
		if verbose:
			print(degrees_array)	
		zero_in=np.where(degrees_array[:,0]==0)[0]
		forbiddens=forbiddens+(zero_in.tolist())
		#3 find node set
		
		while current_degrees[work_node][1] > 0:
			if verbose:
				print("Current degrees :", current_degrees)
				print("idx :", idx)
				print("Current degrees ordered :", current_degrees[idx])
				print("Work-node :", work_node)
				print("Number of outstubs :", current_degrees[work_node][1])
			#3.1
			Li=[]	
			len_li = current_degrees[work_node][1]		
			for i in idx:
				if (i not in forbiddens):
					Li.append(i)
					if (len(Li) == len_li):
						break
			if verbose:
				print("Li :", Li)
			#3.2
			red=np.zeros([n_Vertices])
			for i in range(n_Vertices):
				if( (i not in Li) and (i not in forbiddens)):
					red[i]=1
			#3.3
			Dprime=current_degrees.copy()
			for i in range(len(Li)-1):
				Dprime[Li[i]][0]-=1   
			Dprime[work_node][1]=1
			#3.4  ### update of order not optimal : see paper
			idx_prime = copy.deepcopy(idx)
			for i in range(n_Vertices):
				if idx_prime[i] in Li[:len(Li)-1]:
					if i < n_Vertices - 1 and Dprime[idx_prime[i]][0] < Dprime[idx_prime[i+1]][0]:
						mem = idx_prime[i]
						idx_prime[i] = idx_prime[i+1]
						idx_prime[i+1] = mem 
			# idx_prime = normal_order(Dprime)
			#3.5
			work_node_idx=np.where(np.array(idx_prime) == work_node)[0][0]

			if (work_node_idx!=0):
				k=0		
			else:
				k=1

			if verbose: 
				print("k =", k)

			# Computing G1 for all k
			if verbose:
				print("Dprime ordered :", Dprime[idx_prime].tolist())
			ordered_out = np.asarray(Dprime.tolist())[idx_prime,1]
			ordered_out[0] += 1
			if verbose:
				print("Ordered_out :", ordered_out)
			G1 = [0 for i in range(n_Vertices+2)]
			for i in range(n_Vertices):
				G1[ordered_out[i]] += 1
			if verbose:
				print("G1 :", G1)

			# Computing S for all k
			ordered_out[0] -= 1
			S = [0 for p in range(n_Vertices+2)]
			for t in range(1,n_Vertices): 
				dt = ordered_out[t]
				if t + 1 <= dt: 
					S[dt] -= 1
				if t + 1 <= dt:
					S[dt+1] += 1
			if verbose: 
				print("S :", S)
			# Computing Lprime and Rprime
			Lprime = Dprime[idx_prime[0]][0]

			Gtilde = G1[0] + G1[1]
			Rprime = n_Vertices - 1 - G1[0]  


			if k == 1: 
				Lprime += Dprime[idx_prime[k]][0]
				if Dprime[idx_prime[k]][1] < k+1 :
					Rprime = Rprime + n_Vertices - Gtilde
				else: 
					Rprime = Rprime + n_Vertices - Gtilde - 1
				Gtilde = Gtilde + G1[k+1] + S[k+1]
			
			R_true = 0
			for i in range(k+1):
				R_true += min(k,ordered_out[i])
			for i in range(k+1,n_Vertices): 
				R_true += min(k+1, ordered_out[i])

			while Lprime != Rprime and k < n_Vertices-1:
				k+=1 
				Lprime += Dprime[idx_prime[k]][0] 
				if Dprime[idx_prime[k]][1] < k+1 :
					Rprime = Rprime + n_Vertices - Gtilde
				else: 
					Rprime = Rprime + n_Vertices - Gtilde - 1

				R_true = 0
				for i in range(k+1):
					R_true += min(k,ordered_out[i])
				for i in range(k+1,n_Vertices): 
					R_true += min(k+1, ordered_out[i])

				if R_true != Rprime and verbose:					
					print("R_true :", R_true)
					print("Rprime :", Rprime)
				if k < n_Vertices - 1: 
					Gtilde = Gtilde + G1[k+1] + S[k+1]
				if verbose:
					print("k =", k, ":", Lprime, Rprime)
				# Rprime = R_true

			if Lprime != Rprime: 
				k+=1
			A = []
			if verbose:
				print("k :" , k)
			if k == n_Vertices: 
				for i in range(n_Vertices): 
					if i not in forbiddens: 
						A.append(i)
			else: 
				q = n_Vertices
				for i in idx_prime[k+1:]: 
					if red[i] == 1: 
						q = np.where(np.array(idx) == i)[0][0]
						break
				if verbose: 
					print("q :", q)
				for i in idx[:q]:
					if i not in forbiddens: 
						A.append(i)

			if not A:
				g = construct_simple_graph_HH(degrees)
				if not g.es: 
					return(g)
				else:
					print("Error : can't find a graph with this DD while it exists...") 
					print('Out degrees : ', g.degree(type = 'out'))
					print('In degrees : ', g.degree(type = 'in'))
					plot(g)

			if verbose:
				print("Forbiddens : ", forbiddens)
				print("Red :", red)
				print("idx :", idx)
				print("D :", current_degrees[idx])
				print("idx_prime :", idx_prime)
				print("A : ", A)
			node_to_connect = np.random.choice(A)
			if verbose:
				print('Edge :', (work_node, node_to_connect))
			all_edges.append((work_node, node_to_connect))

			forbiddens.append(node_to_connect)
			current_degrees[node_to_connect][0] -= 1
			current_degrees[work_node][1] -=1

			idx = normal_order(current_degrees)

			if verbose:
				print("\n\n")

		idx = normal_order(current_degrees)
		work_node=idx[work_idx]

	g.add_edges(all_edges)

	return(g)





#######################
#### end		   ####
#######################

#######################




def hamming_distance(g1,g2): 
	Adj1 = np.asarray(list(g1.get_adjacency()))
	Adj2 = np.asarray(list(g2.get_adjacency()))
	return(int(hamming(Adj1.flatten(), Adj2.flatten())*Adj1.shape[0]**2))

def edit_distance(g1,g2):
	Adj1 = np.asarray(list(g1.get_adjacency()))
	Adj2 = np.asarray(list(g2.get_adjacency()))
	return(editdistance.eval(Adj1.flatten(), Adj2.flatten()))

def analyse_graph(g): 
	print("Number of loops :", g.is_loop().count(True))	
	print("Number of multiple edges :", g.is_multiple().count(True))
	print("Vertex connectivity :", g.as_undirected().vertex_disjoint_paths(checks = True))
	print("Edge connectivity :", g.as_undirected().edge_disjoint_paths(checks = True))
	C = g.as_undirected().community_infomap()
	print("Number of connected components :", len(g.as_undirected().clusters().as_cover()))
	plot(C)

def djikstra_distance(g1,g2): 

	d1=np.array(g1.shortest_paths_dijkstra())
	d2=np.array(g2.shortest_paths_dijkstra())
	thresh = np.max(d1[d1 != np.inf])
	d1[d1 > thresh] = thresh
	thresh = np.max(d2[d2 != np.inf])
	d2[d2 > thresh] = thresh
	h1, b1 = np.histogram(d1, bins = int(d1.shape[0]/10))
	h2, b2 = np.histogram(d2, bins = int(d1.shape[0]/10))
	return((np.log(1+(h1-h2)**2 /(h1.shape[0]))).sum())

def pagerank_distance(g1,g2): 
	p1=np.array(g1.pagerank())
	p2=np.array(g2.pagerank())

	h1, b1 = np.histogram(p1, bins = int(p1.shape[0]/10))
	h2, b2 = np.histogram(p2, bins = int(p1.shape[0]/10))
	return((np.log(1+(h1-h2)**2 /(h1.shape[0]))).sum())


def test_distance_swap(n_graphs = 4, n_samples = 100, n_vertices = 100, n_swap_max = 100, sequentially = True, in_distribution = "power", out_distribution = 'power', l = 1, a = 2): 
	n_g = 10
	distances_hamming = np.zeros((n_samples, (n_g+1)*n_graphs, (n_g+1)*n_graphs))
	distances_djikstra = np.zeros((n_samples, (n_g+1)*n_graphs, (n_g+1)*n_graphs))
	distances_pagerank = np.zeros((n_samples, (n_g+1)*n_graphs, (n_g+1)*n_graphs))

	for i in range(n_samples): 
		graphs = []
		for j in range(n_graphs):
			g_init = Graph(directed = True)
			while not g_init.es:
				degrees = generate_degree_sequence(n_vertices, 
												   in_distribution = in_distribution, 
												   out_distribution = out_distribution, 
												   l = l, 
												   a = a)
				g_init = construct_simple_graph_sampling(degrees)

			graphs.append(g_init)
			if sequentially:
				n_swaps = int(n_swap_max/(n_g))
				for k in range(n_g): 
					graphs.append(copy.deepcopy(graphs[j*(n_g+1) + k]))
					swap_random_edges(graphs[j*(n_g+1) + k+1], n_swap = n_swaps)
			else: 
				n_swaps = [int(n_swap_max*(k+1)/(n_g)) for k in range(ng)]
				for k in range(n_g): 
					graphs.append(copy.deepcopy(graphs[j*(n_g+1)]))
					swap_random_edges(graphs[j*(n_g+1) + k+1], n_swap = n_swaps[k])
		k1 = 0
		for g1 in graphs:
			k2 = 0
			for g2 in graphs: 
				distances_hamming[i,k1, k2] = hamming_distance(g1,g2)
				distances_djikstra[i, k1, k2] = djikstra_distance(g1, g2)
				distances_pagerank[i, k1, k2] = pagerank_distance(g1, g2)
				k2 += 1

			k1 += 1

	return(np.mean(distances_hamming, axis = 0), 
		   np.mean(distances_djikstra, axis = 0), 
		   np.mean(distances_pagerank, axis = 0))


def test_time(n_ech = 100): 

	t_HH = 0
	t_config = 0
	t_sampling = 0
	for i in range(n_ech):
		print('Echantillon', i+1)
		degrees = generate_degree_sequence(200, in_distribution = "power", out_distribution = 'power', l = 1, a = 2)
		
		start = time.time()
		g = construct_simple_graph_HH(degrees)
		for j in range(100):
			swap_random_edges(g, 2*len(g.es))
		t_HH += time.time() - start

		start = time.time()
		for j in range(100):
			g = construct_simple_graph_random_edge(degrees)
		t_config += time.time() - start

		start = time.time()
		for j in range(100): 
			g = construct_simple_graph_sampling(degrees)
		t_sampling += time.time() - start 

	return(t_HH/n_ech, t_config/n_ech, t_sampling/n_ech)



if __name__ == '__main__': 

	# g_er = generate_erdos_renyi(50, 0.1)
	# plot(g_er)
	# hist_degrees(g_er)

	# g_smallworld = generate_watts_graph(50, 6, 0.5)
	# plot(g_smallworld)
	# hist_degrees(g_smallworld)

	(t_HH, t_config, t_sampling) = test_time(n_ech = 20)
	print("Execution time Havel-Hakimi :", t_HH)
	print("Execution time configuration model :", t_config)
	print("Execution time sampling model :", t_sampling)

	degrees = np.array([(3,3), (0,1), (2,1), (1,2), (3,2)], dtype = [('in', '<i4'), ('out', '<i4')])
	idx = normal_order(degrees)
	print(degrees[idx])
	degrees = np.array([(3,3), (0,1), (2,1), (1,2), (3,2)], dtype = [('in', '<i4'), ('out', '<i4')])

	g = construct_simple_graph_HH(degrees, verbose = False)
	print('Out degrees : ', g.degree(type = 'out'))
	print('In degrees : ', g.degree(type = 'in'))
	plot(g)
	# analyse_graph(g)

	for i in range(100):
		print("#############################################################################")
		degrees = generate_degree_sequence(10, in_distribution = "power", out_distribution = 'power', l = 1, a = 2)
		g = construct_simple_graph_sampling(degrees, verbose = False)
		# g = construct_simple_graph(degrees)
		# out_degrees = g.degree(type = 'out')
		# in_degrees = g.degree(type = 'in')
		# # print("In_degrees : ")
		# # print(np.array(in_degrees))
		# # print(np.array(degrees.tolist())[:,0])
		# # print("Out_degrees : ")
		# # print(np.array(out_degrees))
		# # print(np.array(degrees.tolist())[:,1])

		# if np.count_nonzero(np.array(in_degrees) - np.array(degrees.tolist())[:,0]) > 0 or np.count_nonzero(np.array(out_degrees) - np.array(degrees.tolist())[:,1]): 
		# 	print("ERROR!!!!!!!!!!!!!!!")

	degrees = generate_degree_sequence(100, in_distribution = "power", out_distribution = 'power', l = 1, a = 2)
	g = construct_simple_graph_HH(degrees, verbose = False)


	degrees = generate_degree_sequence(100, in_distribution = "power", out_distribution = 'power', l = 1, a = 2)
	degrees2 = copy.deepcopy(degrees)
	start = time.time()
	g1 = construct_simple_graph_sampling(degrees, verbose = False)
	g2 = construct_simple_graph_sampling(degrees2, verbose = False)
	end = time.time()
	print("Execution time : ", end - start, "s")
	# print('Out degrees : ', g.degree(type = 'out'))
	# print('In degrees : ', g.degree(type = 'in'))
	# plot(g1)
	# g2 = copy.deepcopy(g1)
	print('Hamming distance :', hamming_distance(g1,g2))
	print('Djikstra distance :', djikstra_distance(g1,g2))
	print('Pagerank distance :', pagerank_distance(g1,g2))
	g3 = copy.deepcopy(g1)
	swap_random_edges(g3,200)
	print('Hamming distance :', hamming_distance(g1,g3))
	print('Djikstra distance :', djikstra_distance(g1,g3))
	print('Pagerank distance :', pagerank_distance(g1,g3))
	print('Hamming distance :', hamming_distance(g2,g3))
	print('Djikstra distance :', djikstra_distance(g3,g2))
	print('Pagerank distance :', pagerank_distance(g3,g2))
	plot(g1)
	plot(g2)

	d_H, d_D, d_P = test_distance_swap(n_samples = 50, 
									   n_graphs = 2, 
									   n_vertices = 100, 
									   n_swap_max = 1000)

	print("d_H :", d_H)
	print("d_D :", d_D)
	print("d_P :", d_P)
	
	plt.figure()
	plt.imshow(d_H)
	plt.colorbar()
	plt.show()

	plt.figure()
	plt.imshow(d_D)
	plt.colorbar()
	plt.show()

	plt.figure()
	plt.imshow(d_P)
	plt.colorbar()
	plt.show()
	# analyse_graph(g1)

	# plt.hist(np.array(g1.pagerank()), bins = 10)
	# plt.show()
	# swap_random_edges(g2,50)
	# # swap_k_random_edges(g2,10, k = 10)

	# # print('Out degrees : ', g.degree(type = 'out'))
	# # print('In degrees : ', g.degree(type = 'in'))
	# plot(g2)
	# analyse_graph(g2)
	# plt.hist(np.array(g2.pagerank()), bins = 10)
	# plt.show()

	# print('Hamming distance :', hamming_distance(g1,g2))
	# # print('Levenshtein distance :', edit_distance(g1,g2))
	# print('Djikstra distance :', djikstra_distance(g1,g2))
	# print('Pagerank distance :', pagerank_distance(g1,g2))

