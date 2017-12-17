import numpy as np 
import scipy.stats as stats
from igraph import *
import time
import copy
from scipy.spatial.distance import hamming

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

def generate_degree_sequence(n_Vertices, distribution = "poisson", l = 5, a = 3):
	# Generates a degree sequence which follows a given distribution

	# Sample degrees from distribution
	if distribution == 'poisson':
		in_degrees = np.random.poisson(lam = l, size = n_Vertices)  
		out_degrees = np.random.poisson(lam = l, size  = n_Vertices) + 1
	elif distribution == 'power':
		in_degrees = truncated_power_law(a = a , N = n_Vertices)  
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

def swap_edges(g, edge1, edge2): 
	new_edge1 = (edge1.tuple[0], edge2.tuple[1])
	new_edge2 = (edge2.tuple[0], edge1.tuple[1])
	g.delete_edges([edge1, edge2])

	g.add_edges([new_edge1, new_edge2])

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
		swap_edges(g, selected_edges[0], selected_edges[1])

def hamming_distance(g1,g2): 
	Adj1 = np.asarray(list(g1.get_adjacency()))
	Adj2 = np.asarray(list(g2.get_adjacency()))
	return(hamming(Adj1.flatten(), Adj2.flatten()))

if __name__ == '__main__': 

	degrees = np.array([(3,3), (0,1), (2,1), (1,2), (3,2)], dtype = [('in', '<i4'), ('out', '<i4')])
	idx = normal_order(degrees)
	print(degrees[idx])

	g = construct_simple_graph(degrees, verbose = True)
	print('Out degrees : ', g.degree(type = 'out'))
	print('In degrees : ', g.degree(type = 'in'))
	plot(g)

	degrees = generate_degree_sequence(100, distribution = "power", a = 2)
	# print(degrees)

	start = time.time()
	g1 = construct_simple_graph(degrees)
	end = time.time()
	print("Execution time : ", end - start, "s")
	# print('Out degrees : ', g.degree(type = 'out'))
	# print('In degrees : ', g.degree(type = 'in'))
	plot(g1)
	print("Is loop?", g1.is_loop().count(True))	
	print("Is multiple?", g1.is_multiple().count(True))

	g2 = copy.deepcopy(g1)

	swap_random_edges(g2,500)
	# print('Out degrees : ', g.degree(type = 'out'))
	# print('In degrees : ', g.degree(type = 'in'))
	plot(g2)
	print("Is loop?", g2.is_loop().count(True))	
	print("Is multiple?", g2.is_multiple().count(True))

	print('Hamming distance :', hamming_distance(g1,g2))
