
#include "../../Storage/intrusive_ptr.hpp"

class Node;

struct Edge{

    pyq_intrusive_ptr<Node> node_fn;



    /*
        Input slot within the parent Node's gradient buffer.

        
    */
    size_t input_nr = 0;


};