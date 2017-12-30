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

def construct_simple_graph(degrees, verbose = False): 
	# Construct a directed with the exact prescribed degree sequence 
	# using Havel-Hakimi greedy algorithm (in the cases it is possible)
	t = 0
	t2 = 0
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

	# Initialize the ordering of the vertices 
	idx = np.array([i for i in range(n_Vertices)])
	current_degrees = degrees
	idx = normal_order(current_degrees)

	all_edges = []

	first_non_zero = 1

	# Loop over the ordered vertices 
	while current_degrees[idx][0][0] != 0: 		

		d_plus = current_degrees[idx][0][0] # out degree of considered vertex
		index = first_non_zero # index of the candidate vertex to link

		# Construct the out-neighbourhood 
		for k in range(d_plus): 
			# Look for the next vertex which still have a non-empty in-neighbourhood
			start = time.time()

			while current_degrees[idx[index]][1] <= 0: 
				index += 1
				# If we reach the end of the vertices list, the degree sequence does not correspond to a simple graph (no loops, multiple edges...)
				if index == n_Vertices: 
					raise RuntimeError("No graph with this DD...")

			end = time.time()
			t2 += end - start 
			if verbose:
				print((idx[0], idx[index]))

			# Add the corresponding edge to the edge set
			all_edges.append(((idx[0], idx[index])))

			# Update the in-degree 
			current_degrees[idx[index]][1] -= 1
			index += 1

			# If we reach the end of the vertices list, the degree sequence does not correspond to a simple graph (no loops, multiple edges...)
			if index == n_Vertices and k < d_plus - 1: 
				raise RuntimeError("No graph with this DD...")
		# Update the out-degree
		current_degrees[idx[0]][0] = 0


		

		index -= 1
		start = time.time()
		if index < n_Vertices - 1 and current_degrees[idx[index]][0] == current_degrees[idx[index+1]][0] and current_degrees[idx[index]][1] < current_degrees[idx[index+1]][1]:
			left_index = index-1
			right_index = index+2
			while left_index >= 0 and current_degrees[idx[left_index]] == current_degrees[idx[index]]:
				left_index-=1
			while right_index < n_Vertices and current_degrees[idx[right_index]] == current_degrees[idx[index+1]]:
				right_index += 1

			left_index += 1
			right_index -= 1
			
			idx_copy = copy.deepcopy(idx)
			
			
			left_part = idx_copy[left_index:index+1]
			right_part = idx_copy[index+1:right_index+1]

			# print(current_degrees[idx])

			# print(current_degrees[left_part])
			# print(current_degrees[right_part])
			# print(current_degrees[idx[left_index:left_index + right_index - index]])
			# print(current_degrees[idx[left_index + right_index - index:right_index + 1]])
			idx[left_index:left_index + right_index - index] = right_part
			idx[left_index + right_index - index:right_index + 1] = left_part
			# print(current_degrees[idx[left_index:left_index + right_index - index]])
			# print(current_degrees[idx[left_index + right_index - index:right_index + 1]])
			# print(current_degrees[idx])

		
		index_to_add = idx[0]
		idx_sorted = idx[1:]

		insert_idx = n_Vertices - 1 - np.searchsorted(current_degrees[idx_sorted[::-1]], current_degrees[index_to_add])
		idx[0:insert_idx] = idx_sorted[0:insert_idx]
		idx[insert_idx] = index_to_add
		idx[insert_idx+1:] = idx_sorted[insert_idx:]
		# print(current_degrees[idx])

		if first_non_zero != 1:
			first_non_zero -= 1
		if first_non_zero > insert_idx and current_degrees[index_to_add][1] > 0: 
			first_non_zero = insert_idx
		
		while first_non_zero < n_Vertices - 1 and current_degrees[idx[first_non_zero]][1] <= 0:
			first_non_zero += 1
		
		# print(first_non_zero)
		# Re-sort the vertices with this new DD
		# idx = normal_order(current_degrees)
		

		# idx_sorted = normal_order(current_degrees)

		# print(current_degrees[idx_sorted])

		# print(current_degrees[idx] == current_degrees[idx_sorted])
		end = time.time()
		t += end - start 
		
	print("Total execution time sorting : ", t, "s")
	print("Execution time checking in-degrees : ", t2, "s")

	g.add_edges(all_edges)

	return(g)

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
		degrees[i][0] = out_degrees[i]
		degrees[i][1] = in_degrees[i]

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



if __name__ == '__main__': 

	g_er = generate_erdos_renyi(50, 0.1)
	plot(g_er)

	degrees = np.array([(3,3), (0,1), (2,1), (1,2), (3,2)], dtype = [('in', '<i4'), ('out', '<i4')])
	idx = normal_order(degrees)
	print(degrees[idx])

	g = construct_simple_graph(degrees, verbose = True)
	print('Out degrees : ', g.degree(type = 'out'))
	print('In degrees : ', g.degree(type = 'in'))
	plot(g)

	degrees = generate_degree_sequence(50, in_distribution = "power", out_distribution = 'power', l = 1, a = 2)

	start = time.time()
	g1 = construct_simple_graph(degrees)
	end = time.time()
	print("Execution time : ", end - start, "s")
	# print('Out degrees : ', g.degree(type = 'out'))
	# print('In degrees : ', g.degree(type = 'in'))
	plot(g1)
	g2 = copy.deepcopy(g1)

	analyse_graph(g1)

	plt.hist(np.array(g1.pagerank()), bins = 10)
	plt.show()
	swap_random_edges(g2,50)
	# swap_k_random_edges(g2,10, k = 10)

	# print('Out degrees : ', g.degree(type = 'out'))
	# print('In degrees : ', g.degree(type = 'in'))
	plot(g2)
	analyse_graph(g2)
	plt.hist(np.array(g2.pagerank()), bins = 10)
	plt.show()

	print('Hamming distance :', hamming_distance(g1,g2))
	# print('Levenshtein distance :', edit_distance(g1,g2))
	print('Djikstra distance :', djikstra_distance(g1,g2))
	print('Pagerank distance :', pagerank_distance(g1,g2))

