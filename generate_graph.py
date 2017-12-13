import numpy as np 
import scipy.stats as stats
from igraph import *
import time

def truncated_power_law(a, N):
	m = N
	x = np.arange(1, m+1, dtype='float')
	pmf = 1/x**a
	pmf /= pmf.sum()
	return(stats.rv_discrete(values=(range(1, m+1), pmf)).rvs(size = N))

def normal_order(degrees): 

	idx = np.argsort(degrees, order = ('in', 'out'))

	return(idx[::-1])

def construct_simple_graph(degrees, verbose = False): 

	g = Graph(directed = True)
	n_Vertices = degrees.shape[0]
	g.add_vertices(n_Vertices)

	sum_in = 0
	sum_out = 0
	for i in range(n_Vertices): 
		sum_out += degrees[i][0]
		sum_in += degrees[i][1]

	if sum_out != sum_in: 
		raise RuntimeError("in and out degrees sum do not match...")

	idx = np.array([i for i in range(n_Vertices)])

	current_degrees = degrees
	idx = normal_order(current_degrees)

	while current_degrees[idx][0][0] != 0: 
		d_plus = current_degrees[idx][0][0]
		index = 1
		for k in range(d_plus): 
			while current_degrees[idx[index]][1] == 0: 
				index += 1
				if index == n_Vertices: 
					raise RuntimeError("No graph with this DD...")
			if verbose:
				print((idx[0], idx[index]))
			g.add_edges([(idx[0], idx[index])])
			current_degrees[idx[index]][1] -= 1
			index += 1
			if index == n_Vertices and k < d_plus - 1: 
				raise RuntimeError("No graph with this DD...")
		current_degrees[idx[0]][0] = 0
		idx = normal_order(current_degrees)

	# print(current_degrees)

	return(g)

def generate_degree_sequence(n_Vertices, distribution = "poisson", l = 5, a = 3):

	if distribution == 'poisson':
		in_degrees = np.random.poisson(lam = l, size = n_Vertices) + 1  
		out_degrees = np.random.poisson(lam = l, size  = n_Vertices) + 1
	if distribution == 'power':
		in_degrees = truncated_power_law(a = a , N = n_Vertices)  
		out_degrees = truncated_power_law(a = a, N = n_Vertices)  

	diff = np.sum(in_degrees) - np.sum(out_degrees)
	for k in range(np.abs(diff)):
		u = np.random.uniform(0,1,size = 1)
		i = np.random.randint(0, n_Vertices, 1)
		if diff > 0: 
			if u > 0.5: 
				in_degrees[i] -= 1
			else: 
				out_degrees[i] += 1
		else: 
			if u > 0.5:
				in_degrees[i] += 1
			else: 
				out_degrees[i] -= 1



	degrees = np.array([(0,0) for k in range(n_Vertices)],  dtype = [('in', '<i4'), ('out', '<i4')])

	for i in range(n_Vertices): 

		degrees[i][0] = out_degrees[i]
		degrees[i][1] = in_degrees[i]

	return(degrees)


if __name__ == '__main__': 

	degrees = np.array([(3,3), (0,1), (2,1), (1,2), (3,2)], dtype = [('in', '<i4'), ('out', '<i4')])
	idx = normal_order(degrees)
	print(degrees[idx])

	g = construct_simple_graph(degrees, verbose = True)
	print('Out degrees : ', g.degree(type = 'out'))
	print('In degrees : ', g.degree(type = 'in'))
	plot(g)

	degrees = generate_degree_sequence(20000, distribution = "power", a = 2)
	# print(degrees)

	start = time.time()
	g = construct_simple_graph(degrees)
	end = time.time()
	print("Execution time : ", end - start, "s")
	# plot(g)