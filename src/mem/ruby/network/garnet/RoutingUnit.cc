/*
 * Copyright (c) 2008 Princeton University
 * Copyright (c) 2016 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "mem/ruby/network/garnet/RoutingUnit.hh"
#include <bits/stdc++.h>
#include <algorithm>
#include "base/cast.hh"
#include "debug/RubyNetwork.hh"
#include "mem/ruby/network/garnet/InputUnit.hh"
#include "mem/ruby/network/garnet/Router.hh"
#include "mem/ruby/slicc_interface/Message.hh"
#include "mem/ruby/network/garnet/flit.hh"

RoutingUnit::RoutingUnit(Router *router)
{
    m_router = router;
	//std::cout<<"ID is "<<m_router->get_id()<<std::endl;
    m_routing_table.clear();
    m_weight_table.clear();
}

void
RoutingUnit::addRoute(std::vector<NetDest>& routing_table_entry)
{
    if (routing_table_entry.size() > m_routing_table.size()) {
        m_routing_table.resize(routing_table_entry.size());
    }
    for (int v = 0; v < routing_table_entry.size(); v++) {
        m_routing_table[v].push_back(routing_table_entry[v]);
    }
}

void
RoutingUnit::addWeight(int link_weight)
{
    m_weight_table.push_back(link_weight);
}

bool
RoutingUnit::supportsVnet(int vnet, std::vector<int> sVnets)
{
    // If all vnets are supported, return true
    if (sVnets.size() == 0) {
        return true;
    }

    // Find the vnet in the vector, return true
    if (std::find(sVnets.begin(), sVnets.end(), vnet) != sVnets.end()) {
        return true;
    }

    // Not supported vnet
    return false;
}

/*
 * This is the default routing algorithm in garnet.
 * The routing table is populated during topology creation.
 * Routes can be biased via weight assignments in the topology file.
 * Correct weight assignments are critical to provide deadlock avoidance.
 */
int
RoutingUnit::lookupRoutingTable(int vnet, NetDest msg_destination)
{
    // First find all possible output link candidates
    // For ordered vnet, just choose the first
    // (to make sure different packets don't choose different routes)
    // For unordered vnet, randomly choose any of the links
    // To have a strict ordering between links, they should be given
    // different weights in the topology file

    int output_link = -1;
    int min_weight = INFINITE_;
    std::vector<int> output_link_candidates;
    int num_candidates = 0;

    // Identify the minimum weight among the candidate output links
    for (int link = 0; link < m_routing_table[vnet].size(); link++) {
        if (msg_destination.intersectionIsNotEmpty(
            m_routing_table[vnet][link])) {

        if (m_weight_table[link] <= min_weight)
            min_weight = m_weight_table[link];
        }
    }

    // Collect all candidate output links with this minimum weight
    for (int link = 0; link < m_routing_table[vnet].size(); link++) {
        if (msg_destination.intersectionIsNotEmpty(
            m_routing_table[vnet][link])) {

            if (m_weight_table[link] == min_weight) {
                num_candidates++;
                output_link_candidates.push_back(link);
            }
        }
    }

    if (output_link_candidates.size() == 0) {
        fatal("Fatal Error:: No Route exists from this Router.");
        exit(0);
    }

    // Randomly select any candidate output link
    int candidate = 0;
    if (!(m_router->get_net_ptr())->isVNetOrdered(vnet))
        candidate = rand() % num_candidates;

    output_link = output_link_candidates.at(candidate);
    return output_link;
}


void
RoutingUnit::addInDirection(PortDirection inport_dirn, int inport_idx)
{
    m_inports_dirn2idx[inport_dirn] = inport_idx;
    m_inports_idx2dirn[inport_idx]  = inport_dirn;
}

void
RoutingUnit::addOutDirection(PortDirection outport_dirn, int outport_idx)
{
    m_outports_dirn2idx[outport_dirn] = outport_idx;
    m_outports_idx2dirn[outport_idx]  = outport_dirn;
}

// outportCompute() is called by the InputUnit
// It calls the routing table by default.
// A template for adaptive topology-specific routing algorithm
// implementations using port directions rather than a static routing
// table is provided here.

int
RoutingUnit::outportCompute(flit *t_flit, int inport,
                            PortDirection inport_dirn)
{
    int outport = -1;
	RouteInfo route = t_flit->get_route();
    if (route.dest_router == m_router->get_id()) {

        // Multiple NIs may be connected to this router,
        // all with output port direction = "Local"
        // Get exact outport id from table
        outport = lookupRoutingTable(route.vnet, route.net_dest);
        return outport;
    }

    // Routing Algorithm set in GarnetNetwork.py
    // Can be over-ridden from command line using --routing-algorithm = 1
    RoutingAlgorithm routing_algorithm =
        (RoutingAlgorithm) m_router->get_net_ptr()->getRoutingAlgorithm();

    switch (routing_algorithm) {
        case TABLE_:  outport =
            lookupRoutingTable(route.vnet, route.net_dest); break;
        case XY_:     outport =
            outportComputeXY(route, inport, inport_dirn); break;
        // any custom algorithm
        case CUSTOM_: outport =
            outportComputeOE(route, inport, inport_dirn); break;
        case 3: outport =
            outportComputeQ_Routing(t_flit, inport, inport_dirn); break;
        default: outport =
            lookupRoutingTable(route.vnet, route.net_dest); break;
    }

    assert(outport != -1);
    return outport;
}

// XY routing implemented using port directions
// Only for reference purpose in a Mesh
// By default Garnet uses the routing table
int
RoutingUnit::outportComputeXY(RouteInfo route,
                              int inport,
                              PortDirection inport_dirn)
{
    PortDirection outport_dirn = "Unknown";

    int M5_VAR_USED num_rows = m_router->get_net_ptr()->getNumRows();
    int num_cols = m_router->get_net_ptr()->getNumCols();
    assert(num_rows > 0 && num_cols > 0);

    int my_id = m_router->get_id();
    int my_x = my_id % num_cols;
    int my_y = my_id / num_cols;

    int dest_id = route.dest_router;
    int dest_x = dest_id % num_cols;
    int dest_y = dest_id / num_cols;

    int x_hops = abs(dest_x - my_x);
    int y_hops = abs(dest_y - my_y);

    bool x_dirn = (dest_x >= my_x);
    bool y_dirn = (dest_y >= my_y);

    // already checked that in outportCompute() function
    assert(!(x_hops == 0 && y_hops == 0));

	std::cout<<inport_dirn<<std::endl;
    if (x_hops > 0) {
        if (x_dirn) {
            assert(inport_dirn == "Local" || inport_dirn == "West");
            outport_dirn = "East";
        } else {
            assert(inport_dirn == "Local" || inport_dirn == "East");
            outport_dirn = "West";
        }
    } else if (y_hops > 0) {
        if (y_dirn) {
            // "Local" or "South" or "West" or "East"
            assert(inport_dirn != "North");
            outport_dirn = "North";
        } else {
            // "Local" or "North" or "West" or "East"
            assert(inport_dirn != "South");
            outport_dirn = "South";
        }
    } else {
        // x_hops == 0 and y_hops == 0
        // this is not possible
        // already checked that in outportCompute() function
        panic("x_hops == y_hops == 0");
    }

    return m_outports_dirn2idx[outport_dirn];
}

//Odd Even routing algorithm

int
RoutingUnit::outportComputeOE(RouteInfo route,
                              int inport,
                              PortDirection inport_dirn)
{
    PortDirection outport_dirn = "Unknown";
    
    int M5_VAR_USED num_rows = m_router->get_net_ptr()->getNumRows();
    int num_cols = m_router->get_net_ptr()->getNumCols();
    assert(num_rows > 0 && num_cols > 0);

    int my_id = m_router->get_id();
    int my_x = my_id % num_cols;
    int my_y = my_id / num_cols;

    int dest_id = route.dest_router;
    int dest_x = dest_id % num_cols;
    int dest_y = dest_id / num_cols;

    int src_id = route.src_router;
    int src_x = src_id % num_cols;
    int src_y = src_id / num_cols;

    int x_hops = dest_x - my_x;
    int y_hops = dest_y - my_y;

    // already checked that in outportCompute() function
    assert(!(x_hops == 0 && y_hops == 0));
	
	if(x_hops == 0) {
		if(y_hops < 0) {
			outport_dirn = "South";
		}
		else {
			outport_dirn = "North";
		}
	}
	else {
		if(x_hops > 0) {
			if(y_hops == 0) {
				outport_dirn = "East";
			}
			else {
				if(my_x % 2 != 0 || my_x == src_x) {
					if(y_hops < 0) {
						outport_dirn = "South";
					}
					else {
						outport_dirn = "North";
					}
				}
				if(dest_x % 2 != 0 || x_hops != 1) {
					outport_dirn = "East";
				}
			}
		}
		else {
			outport_dirn = "West";
			if(y_hops != 0) {
				if(my_x % 2 == 0) {
					if(y_hops < 0) {
						outport_dirn = "South";
					}
					else {
						outport_dirn = "North";
					}
				}
			}	
		}
	}
    return m_outports_dirn2idx[outport_dirn];
}

#define EPSILON 0.3
#define NROUTERS 16
#define NACTIONS 4
#define GRIDSIZE 4
#define LEARNINGRATE 0.5
#define DISCOUNTRATE 0.9


int RoutingUnit::epsilon_greedy(std::vector<std::vector<std::vector<double>>> Q, int state, int destination) {
	float p = (float) rand() / RAND_MAX;
	if(p > EPSILON) {
		int optimalAction = distance(Q[state][destination].begin(), min_element(Q[state][destination].begin(), Q[state][destination].end()));
		std::cout<<"Optimal Action: "<<optimalAction<<std::endl;
		return optimalAction;
	}

	
	int randomAction = rand() % 4;
//	randomAction = temp % 4;
	std::cout<<"Rchd RandomAction: "<<randomAction<<std::endl;
	return randomAction;
}


int RoutingUnit::outportComputeQ_Routing(flit *t_flit, int inport, PortDirection inport_dirn) {
	Tick src_queueing_delay = t_flit->get_src_delay();
    Tick dest_queueing_delay = (curTick() - t_flit->get_dequeue_time());
    Tick queueing_delay = src_queueing_delay + dest_queueing_delay;
	

    PortDirection outport_dirn = "Unknown";
    
	//---Initializing Q-Table---
	static std::vector<std::vector<std::vector<double>>> Q(NROUTERS, std::vector<std::vector<double>>(NROUTERS - 1, std::vector<double> (NACTIONS, 10000)));
	//std::vector<std::vector<std::vector<double>>> Q(NROUTERS, NROUTERS - 1, std::vector<double> (NACTIONS, 0));
	RouteInfo route = t_flit->get_route();
//	std::cout<<"Q Table"<<std::endl;
/*  	for(int i=0; i<NROUTERS; ++i) {
		for(int j=0; j<NROUTERS-1; ++j) {
			for(int k=0; k< 4;++k) {
					std::cout<<Q[i][j][k]<<" ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}	*/
	//---Geting source and destination router details
	int M5_VAR_USED num_rows = m_router->get_net_ptr()->getNumRows();
    int num_cols = m_router->get_net_ptr()->getNumCols();
    assert(num_rows > 0 && num_cols > 0);

    int my_id = m_router->get_id();
    int my_x = my_id % num_cols;
    int my_y = my_id / num_cols;

    int dest_id = route.dest_router;
    int dest_x = dest_id % num_cols;
    int dest_y = dest_id / num_cols;

    int src_id = route.src_router;
    int src_x = src_id % num_cols;
    int src_y = src_id / num_cols;

	int action = epsilon_greedy(Q, my_id, dest_id);
	std::cout<<"Action :"<<action<<std::endl;
	int prev_router_id;
	int temp_x = 0;
	int temp_y = 0;

	std::cout<<"Q_TABLE:  "<<Q[0][0][1]<<std::endl;
	if(inport_dirn == "NORTH"){
		//if(my_y < num_rows - 1) {
			temp_y = my_y + 1;
			temp_x = my_x;
		//}
	}
	else if(inport_dirn == "SOUTH") {
		//if(my_y > 0) {
			temp_y = my_y - 1;
			temp_x = my_x;
		//}
	
	}
	else if(inport_dirn == "EAST") {
		//if(my_x < num_cols -1) {
			temp_x = my_x + 1;
			temp_y = my_y;
		//}
	}
	else {
		//if(my_x > 0) {
			temp_x = my_x - 1;
			temp_y = my_y;
		//}
	
	}

	prev_router_id = temp_y * num_cols + temp_x;

	do{
		std::cout<<"Called "<<action;
		if(action == 0 && my_y < num_rows-1) {
			outport_dirn = "North";
		}
		else if(action == 1 && my_x < num_cols-1) {
			outport_dirn = "East";
		}
		else if(action == 2 && my_y>0) {
			outport_dirn = "South";
		}
		else if(action == 3 && my_x>0){
			outport_dirn = "West";
		}
		else {
			std::cout<<"ELSE CALLED"<<std::endl;
			action = epsilon_greedy(Q, my_id, dest_id);
		}

	}while(outport_dirn == "Unknown");

//	std::cout<<outport_dirn<<std::endl;
	if(my_id == src_id) {
		return m_outports_dirn2idx[outport_dirn];
	}
	int prev_action = 0;
	if(inport_dirn == "North") {
		prev_action = 0;
	}
	else if(inport_dirn == "East") {
		prev_action = 1;
	}
	else if(inport_dirn == "South") {
		prev_action = 2;
	}
	else {
		prev_action = 3;
	}

	int Qy_min = distance(Q[my_id][dest_id].begin(), min_element(Q[my_id][dest_id].begin(), Q[my_id][dest_id].end()));
	
	Q[prev_router_id][dest_id][prev_action] = Q[prev_router_id][dest_id][prev_action] + LEARNINGRATE * (Qy_min + queueing_delay - Q[prev_router_id][dest_id][prev_action]);
	//Update Q_table

    return m_outports_dirn2idx[outport_dirn];
}
// Template for implementing custom routing algorithm
// using port directions. (Example adaptive)
int
RoutingUnit::outportComputeCustom(RouteInfo route,
                                 int inport,
                                 PortDirection inport_dirn)
{
    panic("%s placeholder executed", __FUNCTION__);
}