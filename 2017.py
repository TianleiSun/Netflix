### initalization and loading data
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import prob2utils

data = np.genfromtxt('data.txt', dtype=int)
movie = np.genfromtxt("movies.txt", dtype = "S35", delimiter = "\t")
M = 943
N = 1682
K = 20
rating_mp = {}
for i, j, yij in data:
    if int(j) not in rating_mp:
        rating_mp[int(j)] = []
    rating_mp[int(j)].append(yij)

### Basic Visualizations

# find the most popular movies and best movies
def pop_avg(data):    
    most_pop = sorted(rating_mp, key = lambda k: len(rating_mp[k]), reverse = True) 
    high_rating = sorted(rating_mp, key = lambda k: np.mean(rating_mp[k]), reverse = True)
    return most_pop, high_rating
    
# plot histogram
def pop_avg_hist(data):
    most_pop, high_rating = pop_avg(data)
    ratings_pop = []
    ratings_avg = []
    i = 0
    for movie in most_pop:
        if i == 10: break
        ratings_pop = ratings_pop + rating_mp[movie]
        i += 1
    i = 0
    for movie in high_rating:
        if i == 10: break
        ratings_avg = ratings_avg + rating_mp[movie]
        i += 1
    
    plt.hist(ratings_pop, bins=10, log = False)
    plt.xlabel("rating")
    plt.ylabel("frequency")
    plt.title("Histogram for 10 Most Popular Movies")
    plt.show()
    plt.hist(ratings_avg, bins=10, log = False)
    plt.xlabel("rating")
    plt.ylabel("frequency")
    plt.title("Histogram for 10 Highest-Rating Movies")
    plt.show()

# plot all rating
def all_rating_hist(data):
    plt.hist(data[:, 2], bins=10, log = False)
    plt.xlabel("rating")
    plt.ylabel("frequency")
    plt.title("Histogram for All Movies")
    plt.show() 

# plot 3 genres
def genre_all_hist(data, movie, genre):
    movie_dic = {}
    for m in movie:
        movie_dic[m[0]] = []
        for i in range(2, len(m)):
            movie_dic[m[0]].append(m[i])
    movie_list = []
    rating_list = []
    for m in movie_dic:
        if(movie_dic[m][genre] == '1'):
            movie_list.append(int(m))
    for m in movie_list:
        rating_list = np.hstack((rating_list,rating_mp[m]))
    #plt.hist(rating_list, bins=10, log = False)
    #plt.show()  
    return movie_list, rating_list
#### part 5 SVD plot
def genre_plot(Vt, genre):
    movie_list, rating_list = genre_all_hist(data, movie, genre)
    plt.hist(rating_list, bins=10, log = False)
    plt.xlabel("rating")
    plt.ylabel("frequency")
    plt.title("Histogram for 10 Movies from One Genre")
    plt.show()
    for i in range(10):
        plt.scatter(Vt[0][movie_list[i]], Vt[1][movie_list[i]])
        plt.annotate(movie[movie_list[i]-1][1], (Vt[0][movie_list[i]], Vt[1][movie_list[i]]))
    plt.title('Visualizations for 10 Movies from One Genre Using SVD')
    plt.show()

def SVD_pop_avg_plot(Vt):
    most_pop, high_rating = pop_avg(data)
    # most popular movies
    xlist = []
    ylist = []
    name = []
    i = 0
    for m in most_pop:
        if i == 10: break
        name.append(movie[m - 1, 1])
        #print movie[m - 1]
        xlist.append(Vt[0, m - 1])
        ylist.append(Vt[1, m - 1])
        i += 1
    plt.scatter(xlist, ylist)
    for i, txt in enumerate(name):
        plt.text(xlist[i], ylist[i], txt, fontsize = 8)
    plt.title("Visualization for 10 Most Popular Movies Using SVD")
    plt.show()
    
    # highest rating movies
    xlist = []
    ylist = []
    name = []
    i = 0
    for m in high_rating:
        if i == 10: break
        name.append(movie[m - 1, 1])
        #print movie[m - 1]
        xlist.append(Vt[0, m - 1])
        ylist.append(Vt[1, m - 1])
        i += 1
    plt.scatter(xlist, ylist)
    for i, txt in enumerate(name):
        plt.text(xlist[i], ylist[i], txt, fontsize = 8)
    plt.title("Visualization for 10 Highest-Ranking Movies Using SVD")
    plt.show()

### Matrix Factorization Visualizations
def getUtVt():
    U,V,err = prob2utils.train_model(M,N,K,0.1,0.1,data)
    allratings = np.matmul(U,V)

    A, Si, B = linalg.svd(V)
    A12 = A[:,:2]
    Vt = np.matmul(A12.T,V)
    Ut = np.matmul(A12.T,U.T)
    return Ut, Vt

def testreg():
    reg = [0.0001,0.001,0.01,0.1,1,10]
    err_list = []
    for r in reg:
        U,V,err = prob2utils.train_model(M,N,K,0.1,0.1,data)
        err_list.append(err)
        print err

if __name__ == "__main__":
	data = np.genfromtxt('data.txt', dtype=int)
	movie = np.genfromtxt("movies.txt", dtype = "S35", delimiter = "\t")
	M = 943
	N = 1682
	K = 20
	rating_mp = {}
	for i, j, yij in data:
	    if int(j) not in rating_mp:
	        rating_mp[int(j)] = []
	    rating_mp[int(j)].append(yij)
	pop_avg(data)
	pop_avg_hist(data)
	all_rating_hist(data)
	Ut, Vt = getUtVt()
	SVD_pop_avg_plot(Vt)
	genre_plot(Vt, 4)
