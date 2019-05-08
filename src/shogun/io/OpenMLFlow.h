/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLFLOW_H
#define SHOGUN_OPENMLFLOW_H

#include <shogun/lib/config.h>

#ifdef HAVE_CURL

#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>

#include <curl/curl.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace shogun
{
	/**
	 * Reads OpenML streams which can be downloaded with this function.
	 */
	class OpenMLReader
	{

	public:
		explicit OpenMLReader(const std::string& api_key);

		/**
		 * Returns a string returned by the server given a request.
		 * Raises an error if the returned code is not 200.
		 * Additional arguments can be passed to the request,
		 * which are then concatenated with a "/" character.
		 *
		 * @tparam Args argument type pack, should all be std::string
		 * @param request the request name, see m_request_options
		 * @param format the format to return the data in, see m_format_options
		 * @param args the additional arguments to be passed to request
		 * @return the returned stream from the server if the return code is 200
		 */
		template <typename... Args>
		std::string
		get(const std::string& request, const std::string& format, Args... args)
		{
			m_curl_response_buffer.clear();
			auto find_format = m_format_options.find(format);
			if (find_format == m_format_options.end())
			{
				SG_SERROR(
				    "The provided format \"%s\" is not available\n",
				    format.c_str())
			}
			auto find_request = m_request_options.find(request);
			if (find_request == m_request_options.end())
			{
				SG_SERROR(
				    "Could not find a way to solve the request \"%s\"\n",
				    request.c_str())
			}
			std::string request_format = find_format->second;
			std::string request_path = find_request->second;

			// get additional args and concatenate them with "/"
			if (sizeof...(Args) > 0)
			{
				if (request_path.substr(request_path.size() - 2) == "{}")
				{
					request_path =
					    request_path.substr(0, request_path.size() - 2);
				}
				else
				{
					SG_SERROR(
					    "The provided request \"%s\" cannot handle additional "
					    "args.\n",
					    request.c_str())
				}
				std::vector<std::string> args_vec = {args...};
				std::string args_string = std::accumulate(
				    args_vec.begin() + 1, args_vec.end(), args_vec.front(),
				    [](std::string s0, std::string& s1) {
					    return s0 += "/" + s1;
				    });
				request_path += args_string;
			}

			std::string url = request_format + request_path + "?" + m_api_key;

			openml_curl_request_helper(url);

			return m_curl_response_buffer;
		}

	private:
		/** the raw buffer as a C++ string */
		std::string m_curl_response_buffer;

		/**
		 * Initialises CURL session and gets the data.
		 * This function also handles the response code from the server.
		 *
		 * @param url the url to query
		 */
		void openml_curl_request_helper(const std::string& url);

		/**
		 * Handles all possible codes
		 *
		 * @param curl_handle curl handle used in the request
		 * @param code the code returned by the query
		 */
		void openml_curl_error_helper(CURL* curl_handle, CURLcode code);

		/** the user API key, not required for all requests */
		std::string m_api_key;

		/** the server path to get a response in XML format*/
		static const char* xml_server;
		/** the server path to get a response in JSON format*/
		static const char* json_server;

		/** the server response format options: XML or JSON */
		static const std::unordered_map<std::string, std::string>
		    m_format_options;
		/** all the supported server options */
		static const std::unordered_map<std::string, std::string>
		    m_request_options;

		/* DATA API */
		static const char* dataset_description;
		static const char* list_data_qualities;
		static const char* data_features;
		static const char* data_qualities;
		static const char* list_dataset_qualities;
		static const char* list_dataset_filter;

		/* FLOW API */
		static const char* flow_file;

		/* TASK API */
		static const char* task_file;
	};

	/**
	 * Writes OpenML streams to the OpenML server.
	 */
	class OpenMLWritter
	{
	public:
		OpenMLWritter(const std::string& api_key) : m_api_key(api_key){};

	private:
		/** the user API key, likely to be needed to write to OpenML */
		std::string m_api_key;
	};

	/**
	 * Handles OpenML flows. A flow contains the information
	 * required to instantiate a model.
	 */
	class OpenMLFlow
	{

	public:
		/** alias for component type, map of flows */
		using components_type =
		    std::unordered_map<std::string, std::shared_ptr<OpenMLFlow>>;
		/** alias for parameter type, map of maps with information specific to a
		 * parameter */
		using parameters_type = std::unordered_map<
		    std::string, std::unordered_map<std::string, std::string>>;

		/**
		 * The OpenMLFlow constructor. This constructor is rarely used by the
		 * user and is used by the static class members download_flow and
		 * from_file. The user is expected to use either of the previously
		 * mentioned functions.
		 *
		 * @param name the model name
		 * @param description the model description
		 * @param model the flow class_name field
		 * @param components a map of subflows, i.e. kernels
		 * @param parameters a map of parameter information, i.e. default values
		 * for each parameter name
		 */
		OpenMLFlow(
		    const std::string& name, const std::string& description,
		    const std::string& model, components_type components,
		    parameters_type parameters)
		    : m_name(name), m_description(description), m_class_name(model),
		      m_parameters(parameters), m_components(components)
		{
		}

		/**
		 * Instantiates a OpenMLFlow by downloaded a flow from the OpenML
		 * server.
		 *
		 * @param flow_id the flow ID
		 * @param api_key the user API key (might not be required and can be an
		 * empty string)
		 * @return the OpenMLFlow corresponding to the flow requested
		 * @throws ShogunException when there is a server error or the requested
		 * flow is ill formed.
		 */
		static std::shared_ptr<OpenMLFlow>
		download_flow(const std::string& flow_id, const std::string& api_key);

		/**
		 * Instantiates a OpenMLFlow from a file.
		 * @return the OpenMLFlow corresponding to the flow requested
		 */
		static std::shared_ptr<OpenMLFlow> from_file();

		/**
		 * Publishes a flow to the OpenML server
		 * @param flow the flow to be published
		 */
		static void upload_flow(const std::shared_ptr<OpenMLFlow>& flow);

		/**
		 * Dumps the OpenMLFlow to disk.
		 */
		void dump();

		/**
		 * Gets a subflow, i.e. a kernel in a machine
		 * @param name the name of the subflow, not the flow ID
		 * @return the subflow if it exists
		 */
		std::shared_ptr<OpenMLFlow> get_subflow(const std::string& name)
		{
			auto find_flow = m_components.find(name);
			if (find_flow != m_components.end())
				return find_flow->second;
			else
				SG_SERROR(
				    "The provided subflow could not be found in this flow!")
			return nullptr;
		}

#ifndef SWIG
		SG_FORCED_INLINE parameters_type get_parameters()
		{
			return m_parameters;
		}

		SG_FORCED_INLINE components_type get_components()
		{
			return m_components;
		}

		SG_FORCED_INLINE std::string get_class_name()
		{
			return m_class_name;
		}
#endif // SWIG

	private:
		/** name field of the flow */
		std::string m_name;
		/** description field of the flow */
		std::string m_description;
		/** the class_name field of the flow */
		std::string m_class_name;
		/** the parameter field of the flow (optional) */
		parameters_type m_parameters;
		/** the components fields of the flow (optional) */
		components_type m_components;
	};

	/**
	 * Handles an OpenML dataset.
	 */
	class OpenMLData
	{
	public:
		OpenMLData(
		    const std::string& name, const std::string& description,
		    const std::string& data_format, const std::string& dataset_id,
		    const std::string& version, const std::string& creator,
		    const std::string& contributor, const std::string& collection_date,
		    const std::string& upload_date, const std::string& language,
		    const std::string& license, const std::string& url,
		    const std::string& default_target_attribute,
		    const std::string& row_id_attribute,
		    const std::string& ignore_attribute,
		    const std::string& version_label, const std::string& citation,
		    std::vector<std::string> tag, const std::string& visibility,
		    const std::string& original_data_url, const std::string& paper_url,
		    const std::string& update_comment, const std::string& md5_checksum,
		    std::vector<std::unordered_map<std::string, std::string>>
		        param_descriptors,
		    std::vector<std::unordered_map<std::string, std::string>>
		        param_qualities)

		    : m_name(name), m_description(description),
		      m_data_format(data_format), m_dataset_id(dataset_id),
		      m_version(version), m_creator(creator),
		      m_contributor(contributor), m_collection_date(collection_date),
		      m_upload_date(upload_date), m_language(language),
		      m_license(license), m_url(url),
		      m_default_target_attribute(default_target_attribute),
		      m_row_id_attribute(row_id_attribute),
		      m_ignore_attribute(ignore_attribute),
		      m_version_label(version_label), m_citation(citation),
		      m_tag(std::move(tag)), m_visibility(visibility),
		      m_original_data_url(original_data_url), m_paper_url(paper_url),
		      m_update_comment(update_comment), m_md5_checksum(md5_checksum),
		      m_param_descriptors(std::move(param_descriptors)),
		      m_param_qualities(std::move(param_qualities))
		{
		}

		/**
		 * Creates a dataset instance from a given ID.
		 *
		 */
		static std::shared_ptr<OpenMLData>
		get_data(const std::string& id, const std::string& api_key);

		/**
		 * Returns the dataset
		 * @param api_key
		 * @return
		 */
		std::string get_data_buffer(const std::string& api_key);

	private:
		std::string m_name;
		std::string m_description;
		std::string m_data_format;
		std::string m_dataset_id;
		std::string m_version;
		std::string m_creator;
		std::string m_contributor;
		std::string m_collection_date;
		std::string m_upload_date;
		std::string m_language;
		std::string m_license;
		std::string m_url;
		std::string m_default_target_attribute;
		std::string m_row_id_attribute;
		std::string m_ignore_attribute;
		std::string m_version_label;
		std::string m_citation;
		std::vector<std::string> m_tag;
		std::string m_visibility;
		std::string m_original_data_url;
		std::string m_paper_url;
		std::string m_update_comment;
		std::string m_md5_checksum;
		std::vector<std::unordered_map<std::string, std::string>>
		    m_param_descriptors;
		std::vector<std::unordered_map<std::string, std::string>>
		    m_param_qualities;
	};

	/**
	 * Handles an OpenML split.
	 */
	class OpenMLSplit
	{
	public:
		OpenMLSplit(
		    const std::string& split_id, const std::string& split_type,
		    const std::string& split_url,
		    const std::unordered_map<std::string, std::string>&
		        split_parameters)
		    : m_split_id(split_id), m_split_type(split_type),
		      m_split_url(split_url), m_parameters(split_parameters)
		{
		}

	private:
		std::string m_split_id;
		std::string m_split_type;
		std::string m_split_url;
		std::unordered_map<std::string, std::string> m_parameters;
	};

	/**
	 * Handles OpenML tasks. A task contains all the information
	 * required to train and test a model.
	 */
	class OpenMLTask
	{
	public:
		enum TaskType
		{
			SUPERVISED_CLASSIFICATION = 0,
			SUPERVISED_REGRESSION = 1,
			LEARNING_CURVE = 2,
			SUPERVISED_DATASTREAM_CLASSIFICATION = 3,
			CLUSTERING = 4,
			MACHINE_LEARNING_CHALLENGE = 5,
			SURVIVAL_ANALYSIS = 6,
			SUBGROUP_DISCOVERY = 7
		};
		OpenMLTask(
		    const std::string& task_id, const std::string task_name,
		    TaskType task_type, const std::string& task_type_id,
		    const std::pair<
		        std::shared_ptr<OpenMLData>, std::shared_ptr<OpenMLSplit>>&
		        task_descriptor)
		    : m_task_id(task_id), m_task_name(task_name),
		      m_task_type(task_type), m_task_type_id(task_type_id),
		      m_task_descriptor(task_descriptor)
		{
		}

		static std::shared_ptr<OpenMLTask>
		get_task(const std::string& task_id, const std::string& api_key);

		std::shared_ptr<OpenMLData> get_dataset()
		{
			return m_task_descriptor.first;
		}

	private:
		static TaskType get_task_from_string(const std::string& task_type);

		std::string m_task_id;
		std::string m_task_name;
		TaskType m_task_type;
		std::string m_task_type_id;
		std::pair<std::shared_ptr<OpenMLData>, std::shared_ptr<OpenMLSplit>>
		    m_task_descriptor;
	};

	/**
	 * The Shogun OpenML extension to run models from an OpenMLFlow
	 * and convert models to OpenMLFlow.
	 */
	class ShogunOpenML
	{
	public:
		/**
		 * Instantiates a SGObject from an OpenMLFlow.
		 *
		 * @param flow the flow to instantiate
		 * @param initialize_with_defaults whether to use the default values
		 * specified in the flow
		 * @return the flow as a trainable model
		 */
		static std::shared_ptr<CSGObject> flow_to_model(
		    std::shared_ptr<OpenMLFlow> flow, bool initialize_with_defaults);

		/**
		 * Converts a SGObject to an OpenMLFlow.
		 *
		 * @param model the model to convert
		 * @return the flow from the model conversion
		 */
		static std::shared_ptr<OpenMLFlow>
		model_to_flow(const std::shared_ptr<CSGObject>& model);

	private:
		/**
		 * Helper function to extract module/factory information from the class
		 * name field of OpenMLFlow. Throws an error either if the class name
		 * field is ill formed (i.e. not library.module.algorithm) or if the
		 * library name is not "shogun".
		 *
		 * @param class_name the flow class_name field
		 * @return a tuple with the module name (factory string) and the
		 * algorithm name
		 */
		static std::pair<std::string, std::string>
		get_class_info(const std::string& class_name);
	};
} // namespace shogun
#endif // HAVE_CURL

#endif // SHOGUN_OPENMLFLOW_H
