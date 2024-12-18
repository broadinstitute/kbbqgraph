<h1>KBBQGraph - Barebones and Quick for k-partite graphs</h1>

This is a WIP for a memory-efficient replacement for NetworkX for a specific application.

I work with k-partite graphs when matching datasets, where each dataset is a partite. Most immediately, I need to handle 100s of partites each with 10,000's of nodes, and ~5,000 directed edges connecting each partite set. This quickly saturates NetworkX.

I also need efficient indexing based on partite number and feature number -- this is not offered by any other graphing library.

Someday, I would like to add some very specific homegrown k-partite clustering algorithms. These also assume one-to-one constraints when compressed to undirected, i.e. at most a node _u_ from partite _i_ can only be connected to one node from partite _j_.

License - Free to distribute and use as long as credit to the developers is given.
Also, if you make something better, please tell me.