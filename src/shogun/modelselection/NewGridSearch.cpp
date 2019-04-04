/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/modelselection/NewGridSearch.h>

using namespace shogun;

const std::string ParameterNode::delimiter = "::";

ParameterNode::ParameterNode(CSGObject* model)
{
	m_parent = model;
	SG_REF(model)
	for (auto const& param : model->get_params())
	{
		if (auto* value = m_parent->get(param.first, std::nothrow))
		{
			SG_SPRINT("value=%s\n", value->to_string().c_str())
			create_node(param.first, value);
		}
	}
}

void ParameterNode::create_node(const std::string& name, CSGObject* obj)
{
	SG_SPRINT("ParameterNode::create_node\n")
	m_nodes[name].emplace_back(new ParameterNode(obj));
}

ParameterNode::ParameterNode(
    CSGObject* model, ParameterProperties param_properties)
{
	m_parent = model;
	for (auto const& param : model->get_params())
	{
		if (auto* value = m_parent->get(param.first, std::nothrow))
		{
			m_nodes[param.first].emplace_back(new ParameterNode(value));
			SG_UNREF(value);
		}
	}

	for (auto const& param : model->get_params(param_properties))
	{
		m_param_mapping.insert(
		    std::make_pair(param.first, param.second->get_value()));
	}
}

ParameterNode*
ParameterNode::attach(const std::string& param, ParameterNode* node)
{
	if (!set_param_helper(param, std::make_shared<ParameterNode>(*node)))
		SG_SERROR("Could not attach %s", param.c_str());
	return this;
}

ParameterNode* ParameterNode::attach(
    const std::string& param, const std::shared_ptr<ParameterNode>& node)
{
	if (!set_param_helper(param, node))
		SG_SERROR("Could not attach %s", param.c_str());
	return this;
}

bool ParameterNode::set_param_helper(
    const std::string& param, const std::shared_ptr<ParameterNode>& node)
{
	auto param_iter = param.find(delimiter);

	if (param_iter != std::string::npos)
	{
		return set_param_helper(
		    param.substr(param_iter + delimiter.size()), node);
	}
	else
	{
		if (m_parent->has(param))
		{
			m_nodes[param].push_back(node);
			return true;
		}
	}
	return false;
}

std::string ParameterNode::to_string() const noexcept
{
	std::stringstream ss;
	std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));
	ss << get_name() << "(" << get_parent_name() << "(";

	for (auto param_pair = m_param_mapping.begin();
	     param_pair != m_param_mapping.end(); ++param_pair)
	{
		ss << param_pair->first.c_str() << "=";
		param_pair->second.visit(visitor.get());
		if (std::next(param_pair) != m_param_mapping.end())
			ss << ", ";
	}

	if (!m_nodes.empty())
	{
		if (!m_param_mapping.empty())
			ss << ", ";

		for (auto node = m_nodes.begin(); node != m_nodes.end(); ++node)
		{
			ss << (*node).first << "=";
			if (node->second.size() > 1)
			{
				ss << "[";
			}
			for (auto inner_node = node->second.begin();
			     inner_node != node->second.end(); ++inner_node)
			{
				(*inner_node)->to_string_helper(ss, visitor);
				if (std::next(inner_node) != node->second.end())
					ss << ", ";
			}
			if (node->second.size() > 1)
			{
				ss << "]";
			}
			if (std::next(node) != m_nodes.end())
				ss << ", ";
		}
	}
	ss << "))";
	return ss.str();
}

void ParameterNode::to_string_helper(
    std::stringstream& ss, std::unique_ptr<AnyVisitor>& visitor) const noexcept
{
	ss << get_parent_name() << "(";
	for (auto param_pair = m_param_mapping.begin();
	     param_pair != m_param_mapping.end(); ++param_pair)
	{
		ss << param_pair->first.c_str() << "=";
		param_pair->second.visit(visitor.get());
		if (std::next(param_pair) != m_param_mapping.end())
			ss << ", ";
	}
	ss << ")";
}

bool ParameterNode::set_param_helper(const std::string& param, const Any& value)
{
	auto param_iter = param.find(delimiter);

	if (param_iter != std::string::npos)
	{
		for (auto& node : m_nodes)
		{
			if (node.second.size() > 1)
			{
				SG_SERROR(
				    "Ambiguous call! More than one ParameterNode found in "
				    "%s::%s. To specify a parameter in this ParameterNode use "
				    "the attach method of ParameterNode %s",
				    get_parent_name(), param.c_str(), get_parent_name())
				return false;
			}
			else
				return node.second[0]->set_param_helper(
				    param.substr(param_iter + delimiter.size()), value);
		}
	}
	else
	{
		if (m_parent->has(param))
		{
			m_param_mapping.insert(std::make_pair(param, value));
			return true;
		}
	}

	return false;
}

ParameterNode* ParameterNode::get_current()
{
	auto tree = new ParameterNode(m_parent);

	std::string param;

	auto get_lambda = [&tree, &param](auto val) { tree->attach(param, val); };

	for (const auto& param_pair : m_param_mapping)
	{
		param = param_pair.first;
		sg_any_dispatch(
		    param_pair.second, sg_vector_typemap, shogun::None{}, get_lambda);
	}
	return tree;
}

CSGObject* ParameterNode::to_object(const std::shared_ptr<ParameterNode>& tree)
{
	// TODO: this is incomplete, but i dont want to see compiler warnings for
	//  this
	SG_REF(tree->m_parent)
	return tree->m_parent;
}

void ParameterNode::replace_node(
    const std::string& node_name, size_t index,
    const std::shared_ptr<ParameterNode>& node)
{
	auto it = m_nodes.find(node_name);

	if (it != m_nodes.end())
	{
		if (index < (*it).second.size())
			(*it).second[index] = node;
		else
		{
			SG_SERROR(
			    "Index out of range for %s::%s", get_parent_name(),
			    node_name.c_str())
		}
	}
	else
	{
		SG_SERROR(
		    "Unable to find and replace %s::%s", get_parent_name(),
		    node_name.c_str())
	}
}

GridParameters::GridParameters(CSGObject* model)
    : m_first(true), m_node_complete(false)
{
	m_parent = model;
	SG_REF(model)

	// TODO: once all hyperparameters are flagged properly replace getter with
	//  get_params.model(ParameterProperties::HYPER)
	for (auto const& param : model->get_params())
	{
		if (auto* value = m_parent->get(param.first, std::nothrow))
		{
			SG_SPRINT("value=%s\n", value->to_string().c_str())
			create_node(param.first, value);
			SG_UNREF(value)
			SG_SPRINT("node=%s\n", m_nodes[param.first].back().get()->to_string().c_str())
		}
	}
	SG_SPRINT("GridParameters() - m_parent: %s\n", m_parent->to_string().c_str())
	m_current_node = m_nodes.begin();
	m_current_internal_node = m_current_node->second.begin();
}

void GridParameters::reset()
{
	for (const auto& param_pair : m_param_mapping)
		m_param_iter[param_pair.first] = m_param_begin[param_pair.first];
	m_node_complete = true;
	m_current_param = m_param_mapping.end();
	m_first = true;
}

void GridParameters::create_node(const std::string& name, CSGObject* obj)
{
	m_nodes[name].emplace_back(new GridParameters(obj));
}

ParameterNode* GridParameters::get_next()
{
	SG_SPRINT("GridParameters::get_next() - %s\n", get_parent_name());
	SG_SPRINT("GridParameters::get_next() - %s\n", m_parent->to_string().c_str())
	auto tree = new ParameterNode(m_parent);

	std::string param;

	// Lambda responsible with getting the current value.
	// Also sets m_current_param to the last param in map
	auto get_lambda = [&tree, &param, this](auto val) {
		// decltype(val.begin()) is a "hacky" way of getting the iterator type
		// without registering RandomIterator<T> in type_case.h but need to pass
		// SGVector by value each time
		auto current = any_cast<decltype(val.begin())>(m_param_iter[param]);

		tree->attach(param, *current);

		// if this is not the end node we need to go back to it after fetching
		// the current value
		if (m_current_param != m_param_mapping.end())
		{
			m_current_param = m_param_mapping.end();
		}
	};

	// Lambda responsible with iterating and finding the next parameter to
	// iterate over. Also signals the end of a node
	auto increment_lambda = [&param, this](auto val) {
		auto beginning = any_cast<decltype(val.begin())>(m_param_begin[param]);
		auto current = any_cast<decltype(val.begin())>(m_param_iter[param]);
		auto end = any_cast<decltype(val.begin())>(m_param_end[param]);

		if (std::next(current) != end &&
		    m_current_param != m_param_mapping.end())
		{
			SG_SPRINT("not the end and not the last param: %s\n", param.c_str())
			m_param_iter[param] = make_any(++beginning);
			m_current_param = m_param_mapping.end();
		}
		else if (std::next(current) == end)
		{
			// this is the beginning so need to go back to end of map
			if (m_current_param == m_param_mapping.begin())
			{
				m_param_iter[param] = make_any(beginning);
				SG_SPRINT("the end and the last param: %s\n", param.c_str())
				m_current_param = m_param_mapping.end();
				SG_SPRINT(
				    "updating m_current_param from %s::%s to %s::%s\n",
				    this->get_parent_name(), param.c_str(),
				    this->get_parent_name(), m_current_param->first.c_str())
			}
			// find the next param to iterate over
			else
			{
				bool found_next_param = false;
				auto increment_neighbour = [&param, &found_next_param,
				                            this](auto val) {
					auto beginning =
					    any_cast<decltype(val.begin())>(m_param_begin[param]);
					auto current =
					    any_cast<decltype(val.begin())>(m_param_iter[param]);
					auto end =
					    any_cast<decltype(val.begin())>(m_param_end[param]);

					if (std::next(current) == end)
					{
						SG_SPRINT(
						    "increment_neighbour - end of iteration param: "
						    "%s\n",
						    param.c_str())
						m_param_iter[param] = make_any(beginning);

						if (std::prev(m_current_param) ==
						    m_param_mapping.begin())
						{
							SG_SPRINT(
							    "the end and the last param: %s\n",
							    param.c_str())
							m_current_param = m_param_mapping.end();
							SG_SPRINT(
							    "updating m_current_param from %s::%s to "
							    "%s::%s\n",
							    this->get_parent_name(), param.c_str(),
							    this->get_parent_name(),
							    m_current_param->first.c_str())
							found_next_param = true;
							m_node_complete = true;
						}
						else
						{
							found_next_param = false;
							m_current_param = std::prev(m_current_param);
							param = std::prev(m_current_param)->first;
						}
					}
					else
					{
						SG_SPRINT(
						    "increment_neighbour - not the end of iteration "
						    "param: %s\n",
						    param.c_str())
						m_param_iter[param] = make_any(++current);
						found_next_param = true;
					}
				};
				SG_SPRINT("the end and not the last param: %s\n", param.c_str())
				m_current_param = m_param_mapping.end();
				param = std::prev(m_current_param)->first;
				while (!found_next_param)
				{
					sg_any_dispatch(
					    m_param_mapping[param], sg_vector_typemap,
					    shogun::None{}, increment_neighbour);
				}
				SG_SPRINT(
				    "node is done updating m_current_param to "
				    "%s::%s\n",
				    this->get_parent_name(), param.c_str(),
				    this->get_parent_name(),
				    std::prev(m_current_param)->first.c_str())
			}
		}
		else
		{
			SG_SPRINT("not the end: %s\n", param.c_str())
			m_param_iter[param] = make_any(++current);
		}
	};

	SG_SPRINT("FIRST 1: %s\n", tree->to_string().c_str());

	for (const auto& node : m_nodes)
	{
		std::shared_ptr<ParameterNode> this_node(nullptr);

		ParameterNode* (GridParameters::*fp)() = nullptr;

		if (node != *m_current_node)
			fp = &GridParameters::get_current;
		else
			fp = &GridParameters::get_next;

		for (auto inner_node = node.second.begin();
		     inner_node != node.second.end(); ++inner_node)
		{
			// replace the node of the object with current parameter
			// representation
			SG_SPRINT("INNER NODE %s\n", inner_node->get()->get_parent_name())
			if (inner_node == m_current_internal_node)
			{
				this_node = std::shared_ptr<ParameterNode>(
				    (*std::dynamic_pointer_cast<GridParameters>(*inner_node).*
				     fp)());
				break;
			}
		}
		// need to check if this is an attach or replace, there are
		// situations where the tree isn't aware at this point that it
		// needs nodes, i.e. node is attached after calling constructor
		if (tree->n_nodes() > 0)
			tree->replace_node(node.first, 0, this_node);
		else
			tree->attach(node.first, this_node);
	}

	SG_SPRINT("FIRST 2: %s\n", tree->to_string().c_str());

	if (m_first)
	{
		m_current_param = m_param_mapping.end();
		m_first = false;
	}

	bool iterate_this_loop = false;

	for (auto param_pair = m_param_mapping.begin();
	     param_pair != m_param_mapping.end(); ++param_pair)
	{
		param = param_pair->first;

		SG_SPRINT(
		    "GridParameters::get_next - param: %s, current_param: %s\n",
		    param.c_str(), m_current_param->first.c_str())

		if (m_nodes.empty())
		{
			// check if need to iterate bottom node param
			iterate_this_loop = std::next(param_pair) == m_current_param;
		}
		else
		{
			// check if need to iterate params of this node or child nodes
			iterate_this_loop = std::dynamic_pointer_cast<GridParameters>(
			                        *m_current_internal_node)
			                        ->check_child_node_done() &&
			                    std::next(m_current_internal_node) ==
			                        m_current_node->second.end() &&
			                    std::next(param_pair) == m_current_param;
		}

		if (iterate_this_loop)
		{
			sg_any_dispatch(
			    param_pair->second, sg_vector_typemap, shogun::None{},
			    get_lambda);
			sg_any_dispatch(
			    param_pair->second, sg_vector_typemap, shogun::None{},
			    increment_lambda);
			// we know that nothing else can be iterated so just use a nested
			// loop to avoid any further dynamic casting
			for (; param_pair != m_param_mapping.end(); ++param_pair)
			{
				param = param_pair->first;
				sg_any_dispatch(
				    param_pair->second, sg_vector_typemap, shogun::None{},
				    get_lambda);
			}
			break;
		}
		else
			sg_any_dispatch(
			    param_pair->second, sg_vector_typemap, shogun::None{},
			    get_lambda);
	}

	if (m_nodes.empty())
	{
		if (m_node_complete)
			reset();
	}
	else if (m_current_node != m_nodes.end() && !m_nodes.empty())
	{
		// iterated over the parameter of this node so can set
		// m_current_internal_node to false
		SG_SPRINT(
		    "DEBUG CHECK NODE IS DONE: %s\n",
		    (*m_current_internal_node)->get_parent_name())
		bool node_is_done = std::dynamic_pointer_cast<GridParameters>(
		                        (*m_current_internal_node))
		                        ->check_child_node_done();
		std::dynamic_pointer_cast<GridParameters>(*m_current_internal_node)
		    ->set_node_complete(false);
		if (node_is_done)
		{
			SG_SPRINT(
			    "DEBUG NODE IS DONE: %s\n",
			    (*m_current_internal_node)->get_parent_name())
			++m_current_internal_node;
		}

		if (m_current_internal_node == m_current_node->second.end())
		{
			if (std::next(m_current_node) != m_nodes.end())
				++m_current_node;
			else
				m_current_node = m_nodes.begin();
			m_current_internal_node = m_current_node->second.begin();
			SG_SPRINT(
			    "CURRENT NODE IS: %s \n",
			    (*m_current_internal_node)->get_parent_name())
		}
	}

	return tree;
}

bool GridParameters::set_param_helper(
    const std::string& param, const Any& value)
{
	auto param_iter = param.find(delimiter);

	auto set_iter_lambda = [param, &m_param_begin = m_param_begin,
	                        &m_param_iter = m_param_iter,
	                        &m_param_end = m_param_end](auto val) {
		m_param_begin.insert(std::make_pair(param, make_any(val.begin())));
		m_param_iter.insert(std::make_pair(param, make_any(val.begin())));
		m_param_end.insert(std::make_pair(param, make_any(val.end())));
	};

	if (param_iter != std::string::npos)
	{
		for (auto& node : m_nodes)
		{
			if (node.second.size() > 1)
			{
				SG_SERROR(
				    "Ambiguous call! More than one ParameterNode found in "
				    "%s::%s. To specify a parameter in this ParameterNode use "
				    "the attach method of ParameterNode %s",
				    get_parent_name(), param.c_str(), get_parent_name())
				return false;
			}
			else
				return std::dynamic_pointer_cast<GridParameters>(node.second[0])
				    ->set_param_helper(
				        param.substr(param_iter + delimiter.size()), value);
		}
	}
	else
	{
		if (m_parent->has(param))
		{
			m_param_mapping.insert(std::make_pair(param, value));
			sg_any_dispatch(
			    value, sg_all_typemap, shogun::None{}, set_iter_lambda);
			return true;
		}
	}
	return false;
}

bool GridParameters::is_complete()
{
	if (check_child_node_done())
	{
		for (const auto& node : m_nodes)
		{
			if (!is_inner_complete(node.first))
				return false;
		}
	}
	else
		return false;
	return true;
}

bool GridParameters::is_inner_complete(const std::string& node_name)
{
	for (const auto& inner_node : m_nodes[node_name])
	{
		if (!std::dynamic_pointer_cast<GridParameters>(inner_node)
		         ->check_child_node_done())
			return false;
	}
	return true;
}

ParameterNode* GridParameters::get_current()
{
	auto tree = new ParameterNode(m_parent);

	std::string param;

	auto get_lambda = [&tree, &param, &m_param_iter = m_param_iter](auto val) {
		tree->attach(
		    param, *any_cast<decltype(val.begin())>(m_param_iter[param]));
	};

	for (auto& param_pair : m_param_mapping)
	{
		param = param_pair.first;
		sg_any_dispatch(
		    param_pair.second, sg_vector_typemap, shogun::None{}, get_lambda);
	}
	return tree;
}

GridParameters*
GridParameters::attach(const std::string& param, GridParameters* node)
{
	if (!ParameterNode::set_param_helper(
	        param, std::make_shared<GridParameters>(*node)))
		SG_SERROR("Could not attach %s", param.c_str());
	// Updating node list (might) require resetting the node iterators
	m_current_node = m_nodes.begin();
	m_current_internal_node = m_current_node->second.begin();
	return this;
}

void GridSearch::train()
{
	ASSERT(m_parent != nullptr)
	while (!is_complete())
	{
		auto next = std::shared_ptr<ParameterNode>(get_next());
		auto* machine = ParameterNode::to_object(next)->as<CMachine>();
		SG_SPRINT("CURRENT MACHINE: %s", machine->to_string().c_str())
		machine->train();
		SG_UNREF(machine);
	}
}
