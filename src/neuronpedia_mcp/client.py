#!/usr/bin/env python3

import os
from typing import Optional

import httpx

from .models import AttributionGraphResponse


class NeuronpediaClient:
    def __init__(self, api_key: str, timeout: float = 120.0):
        self.api_key = api_key
        self.base_url = "https://www.neuronpedia.org/api"
        self.client = httpx.AsyncClient(
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            timeout=timeout,
        )

    async def generate_attribution_graph(
        self,
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
    ) -> AttributionGraphResponse:
        data = {
            "prompt": prompt,
            "modelId": model_id,
            "slug": slug,
        }

        if source_set_name is not None:
            data["sourceSetName"] = source_set_name
        if max_n_logits is not None:
            data["maxNLogits"] = max_n_logits
        if desired_logit_prob is not None:
            data["desiredLogitProb"] = desired_logit_prob
        if node_threshold is not None:
            data["nodeThreshold"] = node_threshold
        if edge_threshold is not None:
            data["edgeThreshold"] = edge_threshold
        if max_feature_nodes is not None:
            data["maxFeatureNodes"] = max_feature_nodes
        if qk_top_fraction is not None:
            data["qkTopFraction"] = qk_top_fraction
        if qk_topk is not None:
            data["qkTopk"] = qk_topk

        response = await self.client.post(f"{self.base_url}/graph/generate", json=data)
        response.raise_for_status()
        return AttributionGraphResponse(**response.json())

    async def get_activations(self, model_id: str, source: str, index: str, custom_text: str) -> dict:
        data = {
            "feature": {"modelId": model_id, "source": source, "index": index},
            "customText": custom_text
        }
        response = await self.client.post(f"{self.base_url}/activation/new", json=data)
        response.raise_for_status()
        return response.json()

    async def get_activations_by_feature(self, model_id: str, source: str, index: str) -> dict:
        data = {"modelId": model_id, "source": source, "index": index}
        response = await self.client.post(f"{self.base_url}/activation/get", json=data)
        response.raise_for_status()
        return response.json()

    async def get_activations_by_source(self, model_id: str, source: str, custom_text: str) -> dict:
        data = {"modelId": model_id, "source": source, "customText": custom_text}
        response = await self.client.post(f"{self.base_url}/activation/source", json=data)
        response.raise_for_status()
        return response.json()

    async def get_feature(self, model_id: str, layer: int, index: int) -> dict:
        response = await self.client.get(f"{self.base_url}/feature/{model_id}/{layer}/{index}")
        response.raise_for_status()
        return response.json()

    async def search_all_features(self, text: str, model_id: str = "gemma-2-2b", source_set: str = "res-jb", num_results: int = 10) -> dict:
        data = {"text": text, "modelId": model_id, "sourceSet": source_set, "numResults": num_results}
        response = await self.client.post(f"{self.base_url}/search-all", json=data)
        response.raise_for_status()
        return response.json()

    async def search_topk_by_token(self, text: str, model_id: str, source: str, num_results: int = 10) -> dict:
        data = {"text": text, "modelId": model_id, "source": source, "numResults": num_results}
        response = await self.client.post(f"{self.base_url}/search-topk-by-token", json=data)
        response.raise_for_status()
        return response.json()

    async def generate_explanation(self, model_id: str, layer: str, index: int, explanation_type: str, explanation_model_name: str) -> dict:
        data = {
            "modelId": model_id,
            "layer": layer,
            "index": index,
            "explanationType": explanation_type,
            "explanationModelName": explanation_model_name
        }
        response = await self.client.post(f"{self.base_url}/explanation/generate", json=data)
        response.raise_for_status()
        return response.json()

    async def search_explanations(self, query: str, model_id: str, layers: list, offset: int = 0) -> dict:
        data = {"query": query, "modelId": model_id, "layers": layers, "offset": offset}
        response = await self.client.post(f"{self.base_url}/explanation/search", json=data)
        response.raise_for_status()
        return response.json()

    async def steer_generation(self, prompt: str, model_id: str, features: list, temperature: float = 0.5, n_tokens: int = 48, freq_penalty: float = 2, seed: int = 16, strength_multiplier: float = 4, steer_method: str = "SIMPLE_ADDITIVE") -> dict:
        data = {
            "prompt": prompt,
            "modelId": model_id,
            "features": features,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier,
            "steer_method": steer_method
        }
        response = await self.client.post(f"{self.base_url}/steer", json=data)
        response.raise_for_status()
        return response.json()

    async def steer_chat(self, default_messages: list, steered_messages: list, model_id: str, features: list, temperature: float = 0.5, n_tokens: int = 48, freq_penalty: float = 2, seed: int = 16, strength_multiplier: float = 4, steer_special_tokens: bool = True, steer_method: str = "SIMPLE_ADDITIVE") -> dict:
        data = {
            "defaultChatMessages": default_messages,
            "steeredChatMessages": steered_messages,
            "modelId": model_id,
            "features": features,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier,
            "steer_special_tokens": steer_special_tokens,
            "steer_method": steer_method
        }
        response = await self.client.post(f"{self.base_url}/steer-chat", json=data)
        response.raise_for_status()
        return response.json()

    async def list_graphs(self) -> dict:
        response = await self.client.get(f"{self.base_url}/graph/list")
        response.raise_for_status()
        return response.json()

    async def delete_graph(self, model_id: str, slug: str) -> dict:
        data = {"modelId": model_id, "slug": slug}
        response = await self.client.post(f"{self.base_url}/graph/delete", json=data)
        response.raise_for_status()
        return response.json()

    async def get_graph_metadata(self, model_id: str, slug: str) -> dict:
        response = await self.client.get(f"{self.base_url}/graph/{model_id}/{slug}")
        response.raise_for_status()
        return response.json()

    async def get_signed_put_url(self, filename: str, content_length: int, content_type: str = "application/json") -> dict:
        data = {"filename": filename, "contentLength": content_length, "contentType": content_type}
        response = await self.client.post(f"{self.base_url}/graph/signed-put", json=data)
        response.raise_for_status()
        return response.json()

    async def save_graph_to_db(self, put_request_id: str) -> dict:
        data = {"putRequestId": put_request_id}
        response = await self.client.post(f"{self.base_url}/graph/save-to-db", json=data)
        response.raise_for_status()
        return response.json()

    async def list_subgraphs(self, model_id: str, slug: str) -> dict:
        data = {"modelId": model_id, "slug": slug}
        response = await self.client.post(f"{self.base_url}/graph/subgraph/list", json=data)
        response.raise_for_status()
        return response.json()

    async def save_subgraph(
        self, model_id: str, slug: str, pinned_ids: list, supernodes: list,
        clerps: list, display_name: str = "",
        pruning_threshold: Optional[float] = None,
        density_threshold: Optional[float] = None,
        overwrite_id: Optional[str] = None
    ) -> dict:
        data = {
            "modelId": model_id,
            "slug": slug,
            "pinnedIds": pinned_ids,
            "supernodes": supernodes,
            "clerps": clerps,
        }
        if display_name:
            data["displayName"] = display_name
        if pruning_threshold is not None:
            data["pruningThreshold"] = pruning_threshold
        if density_threshold is not None:
            data["densityThreshold"] = density_threshold
        if overwrite_id is not None:
            data["overwriteId"] = overwrite_id
        response = await self.client.post(f"{self.base_url}/graph/subgraph/save", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_subgraph(self, subgraph_id: str) -> dict:
        data = {"subgraphId": subgraph_id}
        response = await self.client.post(f"{self.base_url}/graph/subgraph/delete", json=data)
        response.raise_for_status()
        return response.json()

    async def add_bookmark(self, model_id: str, layer: str, index: str) -> dict:
        data = {"modelId": model_id, "layer": layer, "index": index}
        response = await self.client.post(f"{self.base_url}/bookmark/add", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_bookmark(self, model_id: str, layer: str, index: str) -> dict:
        data = {"modelId": model_id, "layer": layer, "index": index}
        response = await self.client.post(f"{self.base_url}/bookmark/delete", json=data)
        response.raise_for_status()
        return response.json()

    async def create_list(self, name: str, description: str = "", test_text: Optional[str] = None) -> dict:
        data = {"name": name, "description": description}
        if test_text:
            data["testText"] = test_text
        response = await self.client.post(f"{self.base_url}/list/new", json=data)
        response.raise_for_status()
        return response.json()

    async def create_list_with_features(self, name: str, features: list, description: str = "", test_text: Optional[str] = None) -> dict:
        data = {"name": name, "features": features, "description": description}
        if test_text:
            data["testText"] = test_text
        response = await self.client.post(f"{self.base_url}/list/new-with-features", json=data)
        response.raise_for_status()
        return response.json()

    async def get_user_lists(self) -> dict:
        response = await self.client.post(f"{self.base_url}/list/list")
        response.raise_for_status()
        return response.json()

    async def get_list_details(self, list_id: str) -> dict:
        data = {"listId": list_id}
        response = await self.client.post(f"{self.base_url}/list/get", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_list(self, list_id: str) -> dict:
        data = {"listId": list_id}
        response = await self.client.post(f"{self.base_url}/list/delete", json=data)
        response.raise_for_status()
        return response.json()

    async def update_list(self, list_id: str, name: str, description: str = "", default_test_text: Optional[str] = None) -> dict:
        data = {"listId": list_id, "name": name, "description": description}
        if default_test_text is not None:
            data["defaultTestText"] = default_test_text
        response = await self.client.post(f"{self.base_url}/list/update", json=data)
        response.raise_for_status()
        return response.json()

    async def edit_list_feature(self, list_id: str, model_id: str, layer: str, index: str, description: str) -> dict:
        data = {"listId": list_id, "modelId": model_id, "layer": layer, "index": index, "description": description}
        response = await self.client.post(f"{self.base_url}/list/edit-feature", json=data)
        response.raise_for_status()
        return response.json()

    async def add_features_to_list(self, list_id: str, features: list) -> dict:
        data = {"listId": list_id, "featuresToAdd": features}
        response = await self.client.post(f"{self.base_url}/list/add-features", json=data)
        response.raise_for_status()
        return response.json()

    async def remove_feature_from_list(self, list_id: str, model_id: str, layer: str, index: str) -> dict:
        data = {"listId": list_id, "modelId": model_id, "layer": layer, "index": index}
        response = await self.client.post(f"{self.base_url}/list/remove", json=data)
        response.raise_for_status()
        return response.json()

    async def create_vector(self, model_id: str, layer_number: int, vector: list, vector_label: str, default_steer_strength: float, hook_type: str = "resid-pre") -> dict:
        data = {
            "modelId": model_id,
            "layerNumber": layer_number,
            "hookType": hook_type,
            "vector": vector,
            "vectorDefaultSteerStrength": default_steer_strength,
            "vectorLabel": vector_label
        }
        response = await self.client.post(f"{self.base_url}/vector/new", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_vector(self, model_id: str, source: str, index: str) -> dict:
        data = {"modelId": model_id, "source": source, "index": index}
        response = await self.client.post(f"{self.base_url}/vector/delete", json=data)
        response.raise_for_status()
        return response.json()

    async def get_vector_details(self, model_id: str, source: str, index: str) -> dict:
        data = {"modelId": model_id, "source": source, "index": index}
        response = await self.client.post(f"{self.base_url}/vector/get", json=data)
        response.raise_for_status()
        return response.json()

    async def list_user_vectors(self) -> dict:
        response = await self.client.post(f"{self.base_url}/vector/list-owned")
        response.raise_for_status()
        return response.json()

    async def score_explanation(self, explanation_id: str, scorer_model: str, scorer_type: str) -> dict:
        data = {"explanationId": explanation_id, "scorerModel": scorer_model, "scorerType": scorer_type}
        response = await self.client.post(f"{self.base_url}/explanation/score", json=data)
        response.raise_for_status()
        return response.json()

    async def delete_explanation(self, explanation_id: str) -> dict:
        response = await self.client.post(f"{self.base_url}/explanation/{explanation_id}/delete")
        response.raise_for_status()
        return response.json()

    async def delete_explanation_score(self, explanation_score_id: str) -> dict:
        response = await self.client.post(f"{self.base_url}/explanation/score/{explanation_score_id}/delete")
        response.raise_for_status()
        return response.json()

    async def search_explanations_all(self, query: str, offset: int = 0) -> dict:
        data = {"query": query, "offset": offset}
        response = await self.client.post(f"{self.base_url}/explanation/search-all", json=data)
        response.raise_for_status()
        return response.json()

    async def search_explanations_by_model(self, query: str, model_id: str, offset: int = 0) -> dict:
        data = {"query": query, "modelId": model_id, "offset": offset}
        response = await self.client.post(f"{self.base_url}/explanation/search-model", json=data)
        response.raise_for_status()
        return response.json()

    async def search_explanations_by_release(self, query: str, release_name: str, offset: int = 0) -> dict:
        data = {"query": query, "releaseName": release_name, "offset": offset}
        response = await self.client.post(f"{self.base_url}/explanation/search-release", json=data)
        response.raise_for_status()
        return response.json()

    async def search_explanations_by_layers(self, query: str, model_id: str, layers: list, offset: int = 0) -> dict:
        data = {"query": query, "modelId": model_id, "layers": layers, "offset": offset}
        response = await self.client.post(f"{self.base_url}/explanation/search", json=data)
        response.raise_for_status()
        return response.json()

    async def create_model(self, model_id: str, layers: int, display_name: str = "", url: Optional[str] = None) -> dict:
        data = {"id": model_id, "layers": layers}
        if display_name:
            data["displayName"] = display_name
        if url:
            data["url"] = url
        response = await self.client.post(f"{self.base_url}/model/new", json=data)
        response.raise_for_status()
        return response.json()

    async def search_all_features_advanced(
        self, model_id: str, source_set: str, text: str,
        selected_layers: list = None, sort_indexes: list = None,
        num_results: int = 50, ignore_bos: bool = False,
        density_threshold: float = -1
    ) -> dict:
        data = {
            "modelId": model_id,
            "sourceSet": source_set,
            "text": text,
            "selectedLayers": selected_layers or [],
            "sortIndexes": sort_indexes or [],
            "numResults": num_results,
            "ignoreBos": ignore_bos,
            "densityThreshold": density_threshold
        }
        response = await self.client.post(f"{self.base_url}/search-all", json=data)
        response.raise_for_status()
        return response.json()

    async def search_topk_by_token_advanced(
        self, model_id: str, source: str, text: str,
        num_results: int = 10, ignore_bos: bool = True,
        density_threshold: float = 0.01
    ) -> dict:
        data = {
            "modelId": model_id,
            "source": source,
            "text": text,
            "numResults": num_results,
            "ignoreBos": ignore_bos,
            "densityThreshold": density_threshold
        }
        response = await self.client.post(f"{self.base_url}/search-topk-by-token", json=data)
        response.raise_for_status()
        return response.json()

    async def steer_chat_advanced(
        self, default_messages: list, steered_messages: list,
        model_id: str, features: list, temperature: float = 0.5,
        n_tokens: int = 48, freq_penalty: float = 2, seed: int = 16,
        strength_multiplier: float = 4, steer_special_tokens: bool = True,
        steer_method: str = "SIMPLE_ADDITIVE"
    ) -> dict:
        data = {
            "defaultChatMessages": default_messages,
            "steeredChatMessages": steered_messages,
            "modelId": model_id,
            "features": features,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier,
            "steer_special_tokens": steer_special_tokens,
            "steer_method": steer_method
        }
        response = await self.client.post(f"{self.base_url}/steer-chat", json=data)
        response.raise_for_status()
        return response.json()

    async def steer_text_advanced(
        self, prompt: str, model_id: str, features: list,
        temperature: float = 0.5, n_tokens: int = 48,
        freq_penalty: float = 2, seed: int = 16,
        strength_multiplier: float = 4,
        steer_method: str = "SIMPLE_ADDITIVE"
    ) -> dict:
        data = {
            "prompt": prompt,
            "modelId": model_id,
            "features": features,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier,
            "steer_method": steer_method
        }
        response = await self.client.post(f"{self.base_url}/steer", json=data)
        response.raise_for_status()
        return response.json()

    async def get_connected_neurons(self, model_id: str, layer: int, index: int, trace_depth: int = 1, trace_k: int = 5) -> dict:
        params = {"modelId": model_id, "layer": layer, "index": index, "traceDepth": trace_depth, "traceK": trace_k}
        response = await self.client.get(f"{self.base_url}/sparsity/connected-neurons", params=params)
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()


neuronpedia_client: Optional[NeuronpediaClient] = None


def get_client() -> NeuronpediaClient:
    global neuronpedia_client
    if neuronpedia_client is None:
        api_key = os.getenv("NEURONPEDIA_API_KEY")
        if not api_key:
            raise ValueError("NEURONPEDIA_API_KEY environment variable is required")
        neuronpedia_client = NeuronpediaClient(api_key)
    return neuronpedia_client
