from typing import Optional

from .client import get_client
from .server import mcp


@mcp.tool()
async def generate_attribution_graph(
    prompt: str,
    slug: str,
    model_id: str = "gemma-2-2b",
    source_set_name: Optional[str] = None,
    max_n_logits: Optional[int] = None,
    desired_logit_prob: Optional[float] = None,
    node_threshold: Optional[float] = None,
    edge_threshold: Optional[float] = None,
    max_feature_nodes: Optional[int] = None,
    qk_top_fraction: Optional[float] = None,
    qk_topk: Optional[int] = None,
) -> str:
    """Generate an attribution graph for analyzing text prompts.

    Args:
        prompt: Text prompt to analyze
        slug: Unique identifier for this graph (lowercase, alphanumeric, underscores, hyphens)
        model_id: Model to use (default: gemma-2-2b)
        source_set_name: Optional source set name for graph generation
        max_n_logits: Maximum number of logits to consider (5-15)
        desired_logit_prob: Desired logit probability threshold (0.6-0.99)
        node_threshold: Node threshold for graph (0.5-1)
        edge_threshold: Edge threshold for graph (0.8-1)
        max_feature_nodes: Maximum feature nodes (3000-10000)
        qk_top_fraction: (Lorsa models only) Fraction of top Lorsa heads for QK tracing
        qk_topk: (Lorsa models only) Number of QK upstream contributors per Lorsa target node
    """
    try:
        client = get_client()
        result = await client.generate_attribution_graph(
            prompt=prompt,
            slug=slug,
            model_id=model_id,
            source_set_name=source_set_name,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            max_feature_nodes=max_feature_nodes,
            qk_top_fraction=qk_top_fraction,
            qk_topk=qk_topk,
        )

        return (f"Attribution Graph Generated Successfully!\n\n"
                f"🔗 **View Graph**: {result.url}\n"
                f"📊 **Nodes**: {result.numNodes}\n"
                f"🔗 **Links**: {result.numLinks}\n"
                f"💾 **Data**: {result.s3url}\n\n"
                f"The graph shows how different parts of the {model_id} model contribute "
                f"to generating each token in your prompt: '{prompt}'")

    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def get_feature_activations(
    custom_text: str,
    model_id: str,
    source: str,
    index: str,
) -> str:
    """Get activation values for a specific feature on given text.

    Args:
        custom_text: Input text to analyze
        model_id: Model identifier (e.g., 'gemma-2-2b')
        source: Source or SAE ID (e.g., '9-res-jb')
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.get_activations(model_id, source, index, custom_text)
        return f"Feature Activations for {model_id} {source} Feature {index}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def get_feature_activations_by_feature(
    model_id: str,
    source: str,
    index: str,
) -> str:
    """Get all activations for a given model, source, and index (no input text needed).

    Args:
        model_id: Model ID
        source: Source or SAE ID (e.g., '9-res-jb')
        index: Index of the feature/latent
    """
    try:
        client = get_client()
        result = await client.get_activations_by_feature(model_id, source, index)
        return f"Activations for {model_id} {source} Feature {index}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def get_all_feature_activations_by_source(
    custom_text: str,
    model_id: str,
    source: str,
) -> str:
    """Get activation values for ALL features in a source/SAE when processing input text.

    Args:
        custom_text: Input text to process (single string or JSON array of strings)
        model_id: Model identifier
        source: Source/SAE ID (e.g., '9-res-jb')
    """
    try:
        client = get_client()
        result = await client.get_activations_by_source(model_id, source, custom_text)
        return f"All Feature Activations in {model_id} {source}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def get_feature_details(
    model_id: str,
    layer: int,
    index: int,
) -> str:
    """Get detailed information about a specific feature.

    Args:
        model_id: Model identifier (e.g., 'gemma-2-2b')
        layer: Layer number
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.get_feature(model_id, layer, index)
        return f"Feature Details for {model_id} Layer {layer} Feature {index}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def search_top_features(
    text: str,
    model_id: str = "gemma-2-2b",
    source_set: str = "res-jb",
    num_results: int = 10,
) -> str:
    """Find the top activating features for given text across the entire model.

    Args:
        text: Input text to analyze
        model_id: Model to search (default: gemma-2-2b)
        source_set: SAE set to search (default: res-jb)
        num_results: Number of top features to return (default: 10)
    """
    try:
        client = get_client()
        result = await client.search_all_features(text, model_id, source_set, num_results)
        return f"Top {num_results} Features for '{text}' in {model_id}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def search_features_by_token(
    text: str,
    model_id: str,
    source: str,
    num_results: int = 10,
) -> str:
    """Find top activating features for each token in the text.

    Args:
        text: Input text to analyze
        model_id: Model to search
        source: Source name (e.g., '6-res-jb')
        num_results: Number of top features per token (default: 10)
    """
    try:
        client = get_client()
        result = await client.search_topk_by_token(text, model_id, source, num_results)
        return f"Top Features by Token for '{text}' in {model_id} ({source}):\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def generate_feature_explanation(
    model_id: str,
    layer: str,
    index: int,
    explanation_type: str,
    explanation_model_name: str,
) -> str:
    """Generate an explanation for what a specific feature detects.

    Args:
        model_id: Model identifier (e.g., 'gemma-2-2b')
        layer: SAE ID or Layer (e.g., '9-res-jb')
        index: Feature index
        explanation_type: Type of explanation (e.g., 'oai_attention-head', 'oai_token-act-pair')
        explanation_model_name: Model to use for generation (see Neuronpedia dashboard for supported models)
    """
    try:
        client = get_client()
        result = await client.generate_explanation(model_id, layer, index, explanation_type, explanation_model_name)
        return f"Explanation for {model_id} {layer} Feature {index}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def search_feature_explanations(
    query: str,
    model_id: str,
    layers: str,
    offset: int = 0,
) -> str:
    """Search for explanations in the features of one or more SAEs/Sources in a model.

    Args:
        query: Search query for explanations
        model_id: Model identifier
        layers: JSON array of SAE IDs/layers to search (e.g., '["20-gemmascope-res-16k","21-gemmascope-res-16k"]')
        offset: Pagination offset (default: 0)
    """
    try:
        import json
        client = get_client()
        layers_list = json.loads(layers)
        result = await client.search_explanations(query, model_id, layers_list, offset)
        return f"Feature Explanations for '{query}' in {model_id}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def steer_text_generation(
    prompt: str,
    model_id: str,
    features_json: str,
    temperature: float = 0.5,
    n_tokens: int = 48,
    freq_penalty: float = 2.0,
    seed: int = 16,
    strength_multiplier: float = 4.0,
    steer_method: str = "SIMPLE_ADDITIVE",
) -> str:
    """Steer model text generation using SAE features (non-chat/completions).

    Args:
        prompt: Text prompt to steer
        model_id: Model identifier
        features_json: JSON array of feature objects with modelId, layer, index, strength
        temperature: Generation temperature (default: 0.5)
        n_tokens: Number of tokens to generate (default: 48)
        freq_penalty: Frequency penalty (default: 2.0)
        seed: Random seed (default: 16)
        strength_multiplier: Global strength multiplier (default: 4.0)
        steer_method: Steering method - SIMPLE_ADDITIVE or ORTHOGONAL_DECOMP (default: SIMPLE_ADDITIVE)
    """
    try:
        import json
        client = get_client()
        features = json.loads(features_json)
        result = await client.steer_generation(
            prompt, model_id, features, temperature, n_tokens, freq_penalty, seed, strength_multiplier, steer_method
        )
        return f"Steered Generation Result:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def steer_chat_generation(
    default_messages: str,
    steered_messages: str,
    model_id: str,
    features_json: str,
    temperature: float = 0.5,
    n_tokens: int = 48,
    freq_penalty: float = 2.0,
    seed: int = 16,
    strength_multiplier: float = 4.0,
    steer_special_tokens: bool = True,
    steer_method: str = "SIMPLE_ADDITIVE",
) -> str:
    """Steer chat model generation using SAE features.

    Args:
        default_messages: JSON array of default chat messages
        steered_messages: JSON array of steered chat messages (same format)
        model_id: Model identifier
        features_json: JSON array of feature objects with modelId, layer, index, strength
        temperature: Generation temperature (default: 0.5)
        n_tokens: Number of tokens to generate (default: 48)
        freq_penalty: Frequency penalty (default: 2.0)
        seed: Random seed (default: 16)
        strength_multiplier: Global strength multiplier (default: 4.0)
        steer_special_tokens: Whether to steer special tokens (default: True)
        steer_method: Steering method - SIMPLE_ADDITIVE or ORTHOGONAL_DECOMP (default: SIMPLE_ADDITIVE)
    """
    try:
        import json
        client = get_client()
        default = json.loads(default_messages)
        steered = json.loads(steered_messages)
        features = json.loads(features_json)
        result = await client.steer_chat(
            default, steered, model_id, features, temperature, n_tokens, freq_penalty, seed, strength_multiplier, steer_special_tokens, steer_method
        )
        return f"Steered Chat Generation Result:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def list_user_graphs() -> str:
    """List all attribution graphs created by the user.
    """
    try:
        client = get_client()
        result = await client.list_graphs()
        return f"Your Attribution Graphs:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def delete_attribution_graph(model_id: str, slug: str) -> str:
    """Delete an attribution graph you created.

    Args:
        model_id: Model identifier
        slug: Graph slug identifier
    """
    try:
        client = get_client()
        result = await client.delete_graph(model_id, slug)
        return f"Graph deleted successfully: {result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def upload_graph_get_signed_url(
    filename: str,
    content_length: int,
    content_type: str = "application/json",
) -> str:
    """Step 1/2: Get a pre-signed URL to upload a graph file to S3.

    After getting the URL, upload your JSON file with: curl -X PUT -T <file> <url>
    Then call upload_graph_save_to_db with the putRequestId.

    Args:
        filename: Name of the file to upload
        content_length: Size of the file in bytes (min 1024, max 209715200)
        content_type: MIME type (default: application/json)
    """
    try:
        client = get_client()
        result = await client.get_signed_put_url(filename, content_length, content_type)
        return (f"Pre-Signed URL created.\n\n"
                f"Upload your file:\n"
                f"curl -X PUT -T {filename} '{result['url']}'\n\n"
                f"Then call upload_graph_save_to_db with putRequestId: {result['putRequestId']}")
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def upload_graph_save_to_db(put_request_id: str) -> str:
    """Step 2/2: Save graph metadata to database after uploading the file to S3.

    Args:
        put_request_id: The putRequestId from upload_graph_get_signed_url
    """
    try:
        client = get_client()
        result = await client.save_graph_to_db(put_request_id)
        return f"Graph saved to database. Access it at: {result['url']}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def list_subgraphs(model_id: str, slug: str) -> str:
    """List all subgraphs owned by you for a specific graph.

    Args:
        model_id: Model identifier
        slug: Graph slug
    """
    try:
        client = get_client()
        result = await client.list_subgraphs(model_id, slug)
        return f"Subgraphs for {model_id}/{slug}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def save_subgraph(
    model_id: str,
    slug: str,
    pinned_ids: str,
    supernodes: str,
    clerps: str = "[]",
    display_name: str = "",
    overwrite_id: Optional[str] = None,
) -> str:
    """Create or overwrite a subgraph for a graph.

    Args:
        model_id: Model identifier
        slug: Graph slug
        pinned_ids: JSON array of pinned node IDs (e.g., '["2_15681_2","E_2_0"]')
        supernodes: JSON array of supernode groups (e.g., '[["supernode","4_14735_2","19_9180_3"]]')
        clerps: JSON array of clerp connections (default: '[]')
        display_name: Optional display name for the subgraph
        overwrite_id: ID of existing subgraph to overwrite (optional)
    """
    try:
        import json
        client = get_client()
        result = await client.save_subgraph(
            model_id=model_id,
            slug=slug,
            pinned_ids=json.loads(pinned_ids),
            supernodes=json.loads(supernodes),
            clerps=json.loads(clerps),
            display_name=display_name,
            overwrite_id=overwrite_id,
        )
        return f"Subgraph saved! ID: {result.get('subgraphId')}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def delete_subgraph(subgraph_id: str) -> str:
    """Delete a subgraph you own.

    Args:
        subgraph_id: ID of the subgraph to delete
    """
    try:
        client = get_client()
        result = await client.delete_subgraph(subgraph_id)
        return f"Subgraph deleted: {result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def add_feature_bookmark(model_id: str, layer: str, index: str) -> str:
    """Add a feature to your bookmarks.

    Args:
        model_id: Model identifier
        layer: Layer or SAE identifier
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.add_bookmark(model_id, layer, index)
        return f"Bookmark added for {model_id} layer {layer} feature {index}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def remove_feature_bookmark(model_id: str, layer: str, index: str) -> str:
    """Remove a feature from your bookmarks.

    Args:
        model_id: Model identifier
        layer: Layer or SAE identifier
        index: Feature index
    """
    try:
        client = get_client()
        result = await client.delete_bookmark(model_id, layer, index)
        return f"Bookmark removed for {model_id} layer {layer} feature {index}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def create_feature_list(name: str, description: str = "", test_text: Optional[str] = None) -> str:
    """Create a new feature list.

    Args:
        name: Name of the list
        description: Optional description
        test_text: Optional test text for activation visualization
    """
    try:
        client = get_client()
        result = await client.create_list(name, description, test_text)
        return f"Feature list created: {result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def get_user_feature_lists() -> str:
    """Get all feature lists created by the user.
    """
    try:
        client = get_client()
        result = await client.get_user_lists()
        return f"Your Feature Lists:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def update_feature_list(
    list_id: str,
    name: str,
    description: str = "",
    default_test_text: Optional[str] = None,
) -> str:
    """Update an existing feature list's metadata.

    Args:
        list_id: ID of the list to update
        name: New name for the list
        description: New description for the list
        default_test_text: Optional new default test text for the list
    """
    try:
        client = get_client()
        result = await client.update_list(list_id, name, description, default_test_text)
        return f"Feature list updated: {result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def edit_list_feature_description(
    list_id: str,
    model_id: str,
    layer: str,
    index: str,
    description: str,
) -> str:
    """Update the description of a specific feature in a list.

    Args:
        list_id: ID of the list containing the feature
        model_id: Model the feature belongs to
        layer: Layer or SAE ID of the feature
        index: Index of the feature
        description: New description for the feature
    """
    try:
        client = get_client()
        result = await client.edit_list_feature(list_id, model_id, layer, index, description)
        return f"Feature description updated in list {list_id}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def delete_feature_list(list_id: str) -> str:
    """Delete a feature list.

    Args:
        list_id: ID of the list to delete
    """
    try:
        client = get_client()
        result = await client.delete_list(list_id)
        return f"Feature list deleted: {result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def create_steering_vector(
    model_id: str,
    layer_number: int,
    vector_values: str,
    vector_label: str,
    default_steer_strength: float,
    hook_type: str = "resid-pre"
) -> str:
    """Create a new steering vector.

    Args:
        model_id: Model identifier
        layer_number: Layer number (0-based)
        vector_values: JSON array of vector values
        vector_label: Label for the vector
        default_steer_strength: Default steering strength (-100 to 100)
        hook_type: Hook type (default: resid-pre)
    """
    try:
        import json
        client = get_client()
        vector = json.loads(vector_values)
        result = await client.create_vector(model_id, layer_number, vector, vector_label, default_steer_strength, hook_type)
        return f"Steering vector created: {result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def list_user_vectors() -> str:
    """List all vectors created by the user.
    """
    try:
        client = get_client()
        result = await client.list_user_vectors()
        return f"Your Steering Vectors:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def get_vector_details(model_id: str, source: str, index: str) -> str:
    """Get details of a specific vector.

    Args:
        model_id: Model identifier
        source: Source identifier
        index: Vector index
    """
    try:
        client = get_client()
        result = await client.get_vector_details(model_id, source, index)
        return f"Vector Details:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def search_explanations_by_release(
    query: str,
    release_name: str,
    offset: int = 0,
) -> str:
    """Search for explanations within a specific release.

    Args:
        query: Search query (minimum 3 characters)
        release_name: Name of the release to search within
        offset: Pagination offset (default: 0)
    """
    try:
        client = get_client()
        result = await client.search_explanations_by_release(query, release_name, offset)
        return f"Search Results in release '{release_name}' for '{query}':\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def search_all_explanations(query: str, offset: int = 0) -> str:
    """Search explanations across all features on Neuronpedia.

    Args:
        query: Search query
        offset: Pagination offset (default: 0)
    """
    try:
        client = get_client()
        result = await client.search_explanations_all(query, offset)
        return f"Search Results for '{query}':\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def search_explanations_by_model_advanced(query: str, model_id: str, offset: int = 0) -> str:
    """Search explanations within a specific model.

    Args:
        query: Search query (minimum 3 characters)
        model_id: Model to search within
        offset: Pagination offset (default: 0)
    """
    try:
        client = get_client()
        result = await client.search_explanations_by_model(query, model_id, offset)
        return f"Search Results in {model_id} for '{query}':\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def advanced_feature_search(
    model_id: str,
    source_set: str,
    text: str,
    selected_layers: str = "[]",
    sort_indexes: str = "[]",
    num_results: int = 50,
    ignore_bos: bool = False,
    density_threshold: float = -1
) -> str:
    """Advanced feature search with multiple parameters.

    Args:
        model_id: Model to search
        source_set: SAE set to search
        text: Text to analyze
        selected_layers: JSON array of layer IDs to search (default: all)
        sort_indexes: JSON array of token indexes to sort by (default: max activation)
        num_results: Max results to return (default: 50, max: 100)
        ignore_bos: Don't return results where top activation is BOS token
        density_threshold: Don't return features above this density (0-1, -1 = no threshold)
    """
    try:
        import json
        client = get_client()
        layers = json.loads(selected_layers)
        indexes = json.loads(sort_indexes)
        result = await client.search_all_features_advanced(
            model_id, source_set, text, layers, indexes, num_results, ignore_bos, density_threshold
        )
        return f"Advanced Search Results for '{text}' in {model_id}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def score_feature_explanation(explanation_id: str, scorer_model: str, scorer_type: str) -> str:
    """Score an explanation using AI models.

    Args:
        explanation_id: ID of explanation to score
        scorer_model: Model to use for scoring (e.g., gpt-4o-mini, gemini-1.5-flash)
        scorer_type: Scoring method (recall_alt, eleuther_fuzz, eleuther_recall, eleuther_embedding)
    """
    try:
        client = get_client()
        result = await client.score_explanation(explanation_id, scorer_model, scorer_type)
        return f"Explanation Scoring Result:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def delete_feature_explanation_score(explanation_score_id: str) -> str:
    """Delete an explanation score by its ID.

    Args:
        explanation_score_id: ID of the explanation score to delete
    """
    try:
        client = get_client()
        result = await client.delete_explanation_score(explanation_score_id)
        return f"Explanation score deleted: {result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def advanced_text_steering(
    prompt: str,
    model_id: str,
    features_json: str,
    temperature: float = 0.5,
    n_tokens: int = 48,
    freq_penalty: float = 2.0,
    seed: int = 16,
    strength_multiplier: float = 4.0
) -> str:
    """Advanced text steering with multiple features and parameters.

    Args:
        prompt: Text prompt to steer
        model_id: Model identifier
        features_json: JSON array of feature objects with modelId, layer, index, strength
        temperature: Generation temperature (default: 0.5)
        n_tokens: Number of tokens to generate (default: 48)
        freq_penalty: Frequency penalty (default: 2.0)
        seed: Random seed (default: 16)
        strength_multiplier: Global strength multiplier (default: 4.0)
    """
    try:
        import json
        client = get_client()
        features = json.loads(features_json)
        result = await client.steer_text_advanced(
            prompt, model_id, features, temperature, n_tokens, freq_penalty, seed, strength_multiplier
        )
        return f"Advanced Steering Result:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"


@mcp.tool()
async def get_connected_neurons(
    model_id: str,
    layer: int,
    index: int,
    trace_depth: int = 1,
    trace_k: int = 5,
) -> str:
    """Get connected neurons from the sparsity server, with top explanations.

    Args:
        model_id: Model ID
        layer: Layer index
        index: Neuron index
        trace_depth: Depth of circuit trace (default: 1)
        trace_k: Top K channels/neurons per step in trace (default: 5)
    """
    try:
        client = get_client()
        result = await client.get_connected_neurons(model_id, layer, index, trace_depth, trace_k)
        return f"Connected Neurons for {model_id} layer {layer} neuron {index}:\n\n{result}"
    except Exception as e:
        return f"Error: [{type(e).__name__}] {str(e)}"
