"""Visualization utilities for the perceiver and TAMP structure.

Two main plots:

  plot_dataset_nodes  — scatter of N dataset observations in 2D, colored by
                        which node the perceiver assigns each one.  For the
                        gridworld (obs is already 2-D) the raw (x, y) space is
                        used directly.  Otherwise the g_encoder latent
                        embeddings are projected to 2-D with PCA.

  plot_rollout_nodes  — the same 2-D space, but showing a single rollout
                        trajectory.  Points are colored by node; arrows convey
                        the direction of travel.  The start is marked with a
                        triangle and the goal with a star.

Both functions return the fitted PCA object (or None for gridworld) so the
same projection can be reused across calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import torch
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_dataset_obs(ls, n_obs: int) -> list[np.ndarray]:
    """Return at most n_obs observations sampled uniformly from the dataset."""
    obs_list: list[np.ndarray] = []
    for ep in ls.dataset.iterate_episodes():
        obs_list.extend(ls._extract_obs(ep.observations))
    if len(obs_list) > n_obs:
        idx = np.random.choice(len(obs_list), n_obs, replace=False)
        obs_list = [obs_list[i] for i in idx]
    return obs_list


def _encode_sym(ls, obs_list: list[np.ndarray], batch_size: int = 256) -> np.ndarray:
    """Encode observations with d_net (zero action) → (N, sym_dim) sym embeddings."""
    all_sym = []
    for i in range(0, len(obs_list), batch_size):
        batch = np.stack(obs_list[i : i + batch_size])
        obs_t = ls._preprocess(batch)
        act_t = ls._zero_acts(len(batch))
        with torch.no_grad():
            sym, _ = ls.d_net.encode(obs_t, act_t)
        all_sym.append(sym.cpu().numpy())
    return np.concatenate(all_sym, axis=0)


def _node_colors(K: int) -> list:
    """K distinct colours from tab10/tab20."""
    cmap = plt.get_cmap('tab10' if K <= 10 else 'tab20')
    return [cmap(k / K) for k in range(K)]


def _to_2d(
    ls,
    obs_list: list[np.ndarray],
    pca: PCA | None = None,
    use_pca: bool = True,
) -> tuple[np.ndarray, PCA | None]:
    """Project obs_list into 2-D.

    For gridworld (obs_shape == (2,)) the raw observations are used directly.
    Otherwise:
      - use_pca=True  (default): g_encoder embeddings projected with PCA.
        If pca is provided it is reused; otherwise a new one is fitted.
      - use_pca=False: first two dimensions of the g_encoder embedding used directly.
    """
    if ls.obs_shape == (2,) and not use_pca:
        return np.stack(obs_list).astype(np.float32), None

    sym_embs = _encode_sym(ls, obs_list)

    if not use_pca:
        return sym_embs[:, :2], None

    if pca is None:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(sym_embs)
    else:
        coords = pca.transform(sym_embs)
    return coords, pca


# ---------------------------------------------------------------------------
# Artifact persistence (perceiver + graph)
# ---------------------------------------------------------------------------

def save_pipeline_artifacts(
    output_dir,
    perceiver,
    graph: dict,
    node_state_pool: list[list[np.ndarray]],
    edge_stats: dict | None = None,
) -> None:
    """Save perceiver embeddings, graph structure, and node state pool to output_dir.

    The perceiver closure itself is not picklable, but all data needed to
    reconstruct it (node sym/asym embeddings, obs, states) is saved.
    Pi functions are stripped from the graph — only edge costs are kept.
    """
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir

    torch.save({
        'node_embeddings': perceiver.node_embeddings,  # (K, sym_dim)
        'node_asym':       perceiver.node_asym,        # (K, asym_dim)
        'node_obs':        perceiver.node_obs,          # (K, ...)
        'node_states':     perceiver.node_states,       # (K, ...)
    }, output_dir / 'perceiver_data.pt')

    # Strip non-serialisable pi closures; keep only the numeric fields.
    graph_data = {
        (A, B): (avg_steps, estimated_dist)
        for (A, B), (_, avg_steps, estimated_dist) in graph.items()
    }
    torch.save(graph_data, output_dir / 'graph_data.pt')

    torch.save(node_state_pool, output_dir / 'node_state_pool.pt')

    if edge_stats is not None:
        torch.save(edge_stats, output_dir / 'edge_stats.pt')

    print(f"Saved perceiver_data.pt, graph_data.pt, node_state_pool.pt → {output_dir}")


def load_pipeline_artifacts(output_dir, ls):
    """Load and reconstruct perceiver, graph, and node_state_pool from output_dir.

    Args:
        output_dir: Path to the pipeline output directory.
        ls:         Loaded LearningSystem (needed to reconstruct the perceiver closure).

    Returns:
        perceiver, graph, node_state_pool
    """
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir

    perceiver_data  = torch.load(output_dir / 'perceiver_data.pt',  map_location='cpu', weights_only=False)
    graph_data      = torch.load(output_dir / 'graph_data.pt',       map_location='cpu', weights_only=False)
    node_state_pool = torch.load(output_dir / 'node_state_pool.pt',  map_location='cpu', weights_only=False)
    edge_stats_path = output_dir / 'edge_stats.pt'
    edge_stats      = torch.load(edge_stats_path, map_location='cpu', weights_only=False) if edge_stats_path.exists() else None

    perceiver = _rebuild_perceiver(ls, perceiver_data)
    graph     = _rebuild_graph(graph_data)
    return perceiver, graph, node_state_pool, edge_stats


def _rebuild_perceiver(ls, data: dict):
    """Reconstruct a live perceiver closure from saved node embeddings."""
    node_sym   = data['node_embeddings']
    node_asym  = data['node_asym']
    node_obs   = data['node_obs']
    node_states = data['node_states']

    def perceiver(obs: np.ndarray) -> int:
        obs_t = ls._preprocess(obs[None])
        act_t = ls._zero_acts(1)
        with torch.no_grad():
            sym_q, asym_q = ls.d_net.encode(obs_t, act_t)
        dists = ls._sym_dist_matrix(
            sym_q.cpu().numpy(), asym_q.cpu().numpy(), node_sym, node_asym
        )[0]
        return int(dists.argmin())

    perceiver.node_embeddings = node_sym
    perceiver.node_asym       = node_asym
    perceiver.node_obs        = node_obs
    perceiver.node_states     = node_states
    perceiver.obs_to_state    = {}
    return perceiver


def _rebuild_graph(graph_data: dict) -> dict:
    """Rebuild graph dict with pi=None (sufficient for visualization)."""
    return {
        (A, B): (None, avg_steps, estimated_dist)
        for (A, B), (avg_steps, estimated_dist) in graph_data.items()
    }


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_dataset_nodes(
    ls,
    perceiver,
    n_obs: int = 1000,
    graph: dict | None = None,
    online_graph: dict | None = None,
    use_pca: bool = True,
    node_state_pool: list[list[np.ndarray]] | None = None,
    save_path: str | None = None,
) -> PCA | None:
    """Scatter plot of dataset observations in 2-D, colored by perceived node.

    Args:
        ls:              Trained LearningSystem (must have a dataset if
                         node_state_pool is not provided).
        perceiver:       Output of ls.create_perceiver(K).
        n_obs:           How many observations to sample.
        graph:           Optional output of ls.create_graph(). If provided,
                         edges are drawn as dashed arrows between medoid positions.
        online_graph:    Optional expanded graph from OnlineSystem. Edges present
                         in online_graph but not in graph are drawn as solid orange
                         arrows to highlight which edges were added online.
        use_pca:         If True (default), project sym embeddings with PCA.
                         If False, use the first two latent dimensions directly.
        node_state_pool: Per-node obs lists from ls.create_graph(). When provided,
                         labels are read directly from the pool — no per-obs
                         perceiver calls needed, which is much faster.
        save_path:       If set, the figure is saved here as well as shown.

    Returns:
        The fitted PCA object (or None), so it can be passed to
        plot_rollout_nodes to keep the same projection.
    """
    if node_state_pool is not None:
        # Fast path: labels already known — sample proportionally from each node.
        obs_list, node_ids_list = [], []
        total = sum(len(p) for p in node_state_pool)
        for k, pool in enumerate(node_state_pool):
            if not pool:
                continue
            n_k = max(1, round(n_obs * len(pool) / total))
            idx = np.random.choice(len(pool), min(n_k, len(pool)), replace=False)
            obs_list.extend(pool[i] for i in idx)
            node_ids_list.extend([k] * len(idx))
        node_ids = np.array(node_ids_list)
        print(f"Using node_state_pool: {len(obs_list)} obs across {len(node_state_pool)} nodes.")
    else:
        print(f"Sampling {n_obs} observations from dataset...")
        obs_list = _sample_dataset_obs(ls, n_obs)
        node_ids = np.array([perceiver(o) for o in obs_list])

    coords, pca = _to_2d(ls, obs_list, use_pca=use_pca)
    K = len(perceiver.node_embeddings)
    colors = _node_colors(K)

    use_obs_space = ls.obs_shape == (2,) and not use_pca
    if use_obs_space:
        xlabel, ylabel = 'x', 'y'
        coord_label = 'observation space'
    elif use_pca:
        xlabel, ylabel = 'PCA 1', 'PCA 2'
        coord_label = 'sym embedding (PCA)'
    else:
        xlabel, ylabel = 'sym dim 0', 'sym dim 1'
        coord_label = 'sym embedding (dims 0-1)'

    fig, ax = plt.subplots(figsize=(8, 7))

    for k in range(K):
        mask = node_ids == k
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[k]], s=18, alpha=0.55, label=f'node {k}',
        )

    # Medoid positions in 2-D for star markers and graph edge arrows.
    if use_obs_space:
        medoid_2d = perceiver.node_obs.astype(np.float32)       # (K, 2)
    elif use_pca:
        medoid_2d = pca.transform(perceiver.node_embeddings)    # (K, 2) — node_embeddings = node_sym
    else:
        medoid_2d = perceiver.node_embeddings[:, :2]            # (K, 2)

    for k in range(K):
        ax.scatter(*medoid_2d[k], c=[colors[k]], s=300, marker='*',
                   edgecolors='black', linewidths=1.2, zorder=5)

    # Draw reliable graph edges as dashed arrows between medoid positions.
    if graph is not None:
        for (A, B), edge in graph.items():
            x0, y0 = medoid_2d[A]
            x1, y1 = medoid_2d[B]
            arrow = mpatches.FancyArrowPatch(
                posA=(x0, y0), posB=(x1, y1),
                arrowstyle='->', color=colors[A], lw=1.4,
                linestyle='dashed',
                connectionstyle='arc3,rad=0.15',
                mutation_scale=14,
                zorder=4,
                alpha=0.85,
            )
            ax.add_patch(arrow)

            # Label the edge with estimated_dist at the midpoint (offset perpendicular
            # to the arc to avoid overlapping the arrow).
            estimated_dist = edge[2]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            length = max(np.hypot(dx, dy), 1e-9)
            # Perpendicular unit vector, scaled to a small offset.
            px, py = -dy / length, dx / length
            offset = 0.04 * max(
                np.ptp(medoid_2d[:, 0]) if len(medoid_2d) > 1 else 1.0,
                np.ptp(medoid_2d[:, 1]) if len(medoid_2d) > 1 else 1.0,
            )
            ax.text(
                mx + px * offset, my + py * offset,
                f'{estimated_dist:.1f}',
                fontsize=7, color=colors[A],
                ha='center', va='center', zorder=6,
            )

    # Draw online-only edges (present in online_graph but not in graph) as solid
    # orange arrows so they stand out from the offline dashed edges.
    if online_graph is not None:
        offline_keys = set(graph.keys()) if graph is not None else set()
        new_edges = {k: v for k, v in online_graph.items() if k not in offline_keys}
        for (A, B), edge in new_edges.items():
            x0, y0 = medoid_2d[A]
            x1, y1 = medoid_2d[B]
            arrow = mpatches.FancyArrowPatch(
                posA=(x0, y0), posB=(x1, y1),
                arrowstyle='->', color='darkorange', lw=2.0,
                linestyle='solid',
                connectionstyle='arc3,rad=0.15',
                mutation_scale=16,
                zorder=5,
                alpha=0.95,
            )
            ax.add_patch(arrow)
            estimated_dist = edge[2]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            length = max(np.hypot(dx, dy), 1e-9)
            px, py = -dy / length, dx / length
            offset = 0.04 * max(
                np.ptp(medoid_2d[:, 0]) if len(medoid_2d) > 1 else 1.0,
                np.ptp(medoid_2d[:, 1]) if len(medoid_2d) > 1 else 1.0,
            )
            ax.text(
                mx + px * offset, my + py * offset,
                f'{estimated_dist:.1f}',
                fontsize=7, color='darkorange', fontweight='bold',
                ha='center', va='center', zorder=7,
            )
        if new_edges:
            ax.plot([], [], color='darkorange', lw=2.0, linestyle='solid',
                    label=f'+{len(new_edges)} online edges')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f'Dataset observations by perceived node\n({coord_label}, N={len(obs_list)}, K={K})',
        fontsize=13,
    )
    ax.legend(fontsize=8, ncol=max(1, K // 5), loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    return pca


def _plot_trajectory(
    ls,
    perceiver,
    obs_list: list[np.ndarray],
    goal_obs: np.ndarray | None,
    title: str,
    pca: PCA | None = None,
    use_pca: bool = True,
    save_path: str | None = None,
) -> None:
    """Shared trajectory plotting logic used by plot_rollout_nodes and plot_demo_nodes."""
    node_ids = np.array([perceiver(o) for o in obs_list])

    coords, _ = _to_2d(ls, obs_list, pca=pca, use_pca=use_pca)
    K = len(perceiver.node_embeddings)
    colors = _node_colors(K)

    use_obs_space = ls.obs_shape == (2,) and not use_pca
    if use_obs_space:
        xlabel, ylabel = 'x', 'y'
    elif use_pca:
        xlabel, ylabel = 'PCA 1', 'PCA 2'
    else:
        xlabel, ylabel = 'latent dim 0', 'latent dim 1'

    fig, ax = plt.subplots(figsize=(8, 7))

    # Arrows between consecutive steps, colored by the source node.
    for t in range(len(coords) - 1):
        x0, y0 = coords[t]
        x1, y1 = coords[t + 1]
        k = int(node_ids[t])
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='->', color=colors[k], lw=1.5,
                connectionstyle='arc3,rad=0.05',
            ),
        )

    # Points colored by node.
    for t, (x, y) in enumerate(coords):
        ax.scatter(x, y, c=[colors[int(node_ids[t])]], s=40, zorder=3)

    # Start (triangle) and end (square).
    ax.scatter(*coords[0],  c='white', s=200, marker='^', edgecolors='black',
               linewidths=1.5, zorder=5)
    ax.scatter(*coords[-1], c='white', s=200, marker='s', edgecolors='black',
               linewidths=1.5, zorder=5)

    # Goal marker (star), if provided.
    goal_node = None
    if goal_obs is not None:
        goal_coords, _ = _to_2d(ls, [goal_obs], pca=pca, use_pca=use_pca)
        goal_node = perceiver(goal_obs)
        ax.scatter(*goal_coords[0], c=[colors[goal_node]], s=350, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=6)

    # Legend.
    seen_nodes = sorted(set(node_ids.tolist()))
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[k],
                   markersize=9, label=f'node {k}')
        for k in seen_nodes
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=10, label='start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=10, label='end'),
    ]
    if goal_node is not None:
        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=colors[goal_node],
                       markeredgecolor='black', markersize=13, label=f'goal (node {goal_node})'),
        )
    ax.legend(handles=legend_handles, fontsize=8, loc='best')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_rollout_nodes(
    ls,
    perceiver,
    result: dict,
    pca: PCA | None = None,
    use_pca: bool = True,
    save_path: str | None = None,
) -> None:
    """Trajectory plot of a policy rollout in 2-D, colored by perceived node."""
    use_obs_space = ls.obs_shape == (2,)
    if use_obs_space:
        coord_label = 'observation space'
    elif use_pca:
        coord_label = 'sym embedding (PCA)'
    else:
        coord_label = 'sym embedding (dims 0-1)'
    success_str = '✓' if result.get('success') else '✗'
    K = len(perceiver.node_embeddings)
    title = (
        f'Rollout trajectory by perceived node  {success_str}\n'
        f'({coord_label}, {len(result["obs"])} steps, K={K})'
    )
    _plot_trajectory(
        ls, perceiver,
        obs_list=result['obs'],
        goal_obs=result.get('goal_obs'),
        title=title,
        pca=pca,
        use_pca=use_pca,
        save_path=save_path,
    )


def plot_loss_curves(
    loss_history: list[dict],
    save_path: str | None = None,
) -> None:
    """Plot training loss curves from the history returned by LearningSystem.train().

    Top panel: critic loss.
    Bottom panel: actor loss broken down into crl / bc / ent components.

    Args:
        loss_history: List of dicts with keys step, critic, actor, crl, bc, ent.
        save_path:    If set, the figure is saved here.
    """
    if not loss_history:
        print("loss_history is empty — nothing to plot.")
        return

    steps  = [d['step']   for d in loss_history]
    critic = [d['critic'] for d in loss_history]
    actor  = [d['actor']  for d in loss_history]
    crl    = [d['crl']    for d in loss_history]
    bc     = [d['bc']     for d in loss_history]
    ent    = [d['ent']    for d in loss_history]

    fig, (ax_c, ax_a) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_c.plot(steps, critic, color='steelblue', linewidth=1.2)
    ax_c.set_ylabel('Critic loss', fontsize=11)
    ax_c.set_title('Training loss curves', fontsize=13)
    ax_c.grid(True, alpha=0.3)

    ax_a.plot(steps, actor, color='black',   linewidth=1.2, label='actor total')
    ax_a.plot(steps, crl,   color='coral',   linewidth=1.0, linestyle='--', label='crl')
    ax_a.plot(steps, bc,    color='seagreen',linewidth=1.0, linestyle='--', label='bc')
    ax_a.plot(steps, ent,   color='orchid',  linewidth=1.0, linestyle='--', label='ent')
    ax_a.set_xlabel('Step', fontsize=11)
    ax_a.set_ylabel('Actor loss', fontsize=11)
    ax_a.legend(fontsize=9, loc='upper right')
    ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_online_loss_curves(
    loss_history: list[dict],
    save_path: str | None = None,
) -> None:
    """Plot online training loss curves (critic + crl/ent actor; no bc term).

    Args:
        loss_history: List of dicts with keys step, critic, actor, crl, ent.
        save_path:    If set, the figure is saved here.
    """
    if not loss_history:
        print("loss_history is empty — nothing to plot.")
        return

    steps  = [d['step']   for d in loss_history]
    critic = [d['critic'] for d in loss_history]
    actor  = [d['actor']  for d in loss_history]
    crl    = [d['crl']    for d in loss_history]
    ent    = [d['ent']    for d in loss_history]

    fig, (ax_c, ax_a) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_c.plot(steps, critic, color='steelblue', linewidth=1.2)
    ax_c.set_ylabel('Critic loss', fontsize=11)
    ax_c.set_title('Online training loss curves', fontsize=13)
    ax_c.grid(True, alpha=0.3)

    ax_a.plot(steps, actor, color='black',  linewidth=1.2, label='actor total')
    ax_a.plot(steps, crl,   color='coral',  linewidth=1.0, linestyle='--', label='crl')
    ax_a.plot(steps, ent,   color='orchid', linewidth=1.0, linestyle='--', label='ent')
    ax_a.set_xlabel('Step', fontsize=11)
    ax_a.set_ylabel('Actor loss', fontsize=11)
    ax_a.legend(fontsize=9, loc='upper right')
    ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_edge_success_rates(
    edge_stats: dict,
    success_threshold: float = 0.5,
    save_path: str | None = None,
) -> None:
    """Histogram of per-edge success rates from create_graph, with the threshold marked.

    Also shows a K×K heatmap of success rates so you can see which node pairs
    are navigable and which are not.

    Args:
        edge_stats:         {(A, B): success_rate} from create_graph.
        success_threshold:  The cutoff used when building the graph (drawn as a
                            vertical line on the histogram).
        save_path:          If set, the figure is saved here.
    """
    if not edge_stats:
        print("edge_stats is empty — nothing to plot.")
        return

    edges   = sorted(edge_stats.keys())
    rates   = [edge_stats[e] for e in edges]
    K       = max(max(a, b) for a, b in edges) + 1

    fig, (ax_hist, ax_heat) = plt.subplots(1, 2, figsize=(12, 4))

    # --- Histogram ---
    ax_hist.hist(rates, bins=np.linspace(0, 1, 21), color='steelblue', edgecolor='white')
    ax_hist.axvline(success_threshold, color='crimson', linestyle='--', linewidth=1.5,
                    label=f'threshold={success_threshold}')
    n_pass = sum(r >= success_threshold for r in rates)
    ax_hist.set_xlabel('Success rate', fontsize=11)
    ax_hist.set_ylabel('Number of edges', fontsize=11)
    ax_hist.set_title(
        f'Edge success rates  ({n_pass}/{len(rates)} edges pass threshold)',
        fontsize=12,
    )
    ax_hist.legend(fontsize=9)

    # --- Heatmap ---
    matrix = np.full((K, K), np.nan)
    for (A, B), r in edge_stats.items():
        matrix[A, B] = r
    im = ax_heat.imshow(matrix, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    ax_heat.set_xticks(range(K))
    ax_heat.set_yticks(range(K))
    ax_heat.set_xlabel('Target node B', fontsize=11)
    ax_heat.set_ylabel('Source node A', fontsize=11)
    ax_heat.set_title('Success rate heatmap  (A → B)', fontsize=12)
    for (A, B), r in edge_stats.items():
        ax_heat.text(B, A, f'{r:.2f}', ha='center', va='center', fontsize=7,
                     color='black' if 0.2 < r < 0.8 else 'white')
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_graph_networkx(
    K: int,
    offline_graph: dict,
    online_graph: dict | None = None,
    save_path: str | None = None,
) -> None:
    """Draw the node/edge graph using networkx with a spring layout.

    Offline edges are blue, online-promoted edges are orange dashed.

    Args:
        K:              Number of nodes.
        offline_graph:  Graph dict from the offline pipeline.
        online_graph:   Expanded graph dict from the online system (optional).
        save_path:      If set, the figure is saved here.
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(K))

    offline_edges = list(offline_graph.keys())
    online_edges = []
    if online_graph is not None:
        online_edges = [k for k in online_graph if k not in offline_graph]

    G.add_edges_from(offline_edges)
    G.add_edges_from(online_edges)

    pos = nx.spring_layout(G, seed=42, k=2.0 / max(K**0.5, 1))

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500,
                           node_color='lightsteelblue', edgecolors='black', linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight='bold')

    # Draw offline edges
    if offline_edges:
        nx.draw_networkx_edges(G, pos, edgelist=offline_edges, ax=ax,
                               edge_color='#4a90d9', width=1.5,
                               arrows=True, arrowsize=15, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.1',
                               label=f'offline ({len(offline_edges)})')

    # Draw online-promoted edges
    if online_edges:
        nx.draw_networkx_edges(G, pos, edgelist=online_edges, ax=ax,
                               edge_color='darkorange', width=2.0, style='dashed',
                               arrows=True, arrowsize=15, arrowstyle='-|>',
                               connectionstyle='arc3,rad=0.1',
                               label=f'online ({len(online_edges)})')

    ax.legend(fontsize=10, loc='upper left')
    ax.set_title(f'Graph  ({K} nodes, {len(offline_edges)} offline + '
                 f'{len(online_edges)} online edges)', fontsize=13)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_node_distances(
    distance_matrix: np.ndarray,
    title: str = 'Node-pair distances',
    save_path: str | None = None,
) -> None:
    """Heatmap of pairwise node distances, log-scaled for readability.

    Args:
        distance_matrix: (K, K) array of distances (inf = unreachable).
        title:           Plot title.
        save_path:       If set, the figure is saved here.
    """
    K = distance_matrix.shape[0]
    d = distance_matrix.copy()
    d[~np.isfinite(d)] = np.nan
    d_log = np.log1p(np.where(np.isnan(d), 0, d))
    d_log[np.isnan(d)] = np.nan

    fig, (ax_raw, ax_log) = plt.subplots(1, 2, figsize=(13, 5))

    # Raw distances
    im1 = ax_raw.imshow(d, aspect='equal', cmap='viridis')
    ax_raw.set_xticks(range(K))
    ax_raw.set_yticks(range(K))
    ax_raw.set_xlabel('Target', fontsize=11)
    ax_raw.set_ylabel('Source', fontsize=11)
    ax_raw.set_title(f'{title} (raw)', fontsize=12)
    for i in range(K):
        for j in range(K):
            if np.isfinite(distance_matrix[i, j]) and i != j:
                ax_raw.text(j, i, f'{distance_matrix[i,j]:.0f}',
                           ha='center', va='center', fontsize=6, color='white')
    fig.colorbar(im1, ax=ax_raw, fraction=0.046, pad=0.04)

    # Log-scaled
    im2 = ax_log.imshow(d_log, aspect='equal', cmap='viridis')
    ax_log.set_xticks(range(K))
    ax_log.set_yticks(range(K))
    ax_log.set_xlabel('Target', fontsize=11)
    ax_log.set_ylabel('Source', fontsize=11)
    ax_log.set_title(f'{title} (log1p)', fontsize=12)
    fig.colorbar(im2, ax=ax_log, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_online_edge_status(
    K: int,
    offline_graph: dict,
    online_graph: dict,
    save_path: str | None = None,
) -> None:
    """K×K heatmap showing edge status: no edge, offline edge, or online-promoted edge.

    Args:
        K:              Number of nodes.
        offline_graph:  Graph dict from the offline pipeline.
        online_graph:   Expanded graph dict from the online system.
        save_path:      If set, the figure is saved here.
    """
    # 0 = no edge, 1 = offline edge, 2 = online-promoted edge
    matrix = np.zeros((K, K), dtype=int)
    for (A, B) in offline_graph:
        matrix[A, B] = 1
    for (A, B) in online_graph:
        if (A, B) not in offline_graph:
            matrix[A, B] = 2

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#e0e0e0', '#4a90d9', '#e8833a'])  # grey, blue, orange
    labels = ['no edge', 'offline', 'online']

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect='equal')

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xlabel('Target node B', fontsize=11)
    ax.set_ylabel('Source node A', fontsize=11)

    n_offline = sum(1 for v in matrix.flat if v == 1)
    n_online  = sum(1 for v in matrix.flat if v == 2)
    ax.set_title(f'Edge status  ({n_offline} offline, {n_online} online)', fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(cmap.colors, labels)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_node_samples(
    ls,
    perceiver,
    n_samples: int = 5,
    node_state_pool: list[list[np.ndarray]] | None = None,
    env=None,
    save_path: str | None = None,
) -> None:
    """Grid of sample observations from each perceiver node.

    Rows = nodes, columns = individual samples.  The medoid (K-medoids centre)
    is always shown in the first column with a coloured border.

    If env is provided, each cell is rendered via env.reset_to() + env.render(),
    giving a real visual frame of what the environment looks like at that state.
    Otherwise, image obs are shown directly and flat obs as bar charts.

    Args:
        ls:              Trained LearningSystem (must have a dataset if
                         node_state_pool is not provided).
        perceiver:       Output of ls.create_perceiver(K).
        n_samples:       Number of samples per node (including the medoid).
        node_state_pool: Per-node obs lists from ls.create_graph().  If None,
                         the dataset is scanned and classified on the fly.
        env:             Optional environment.  When provided, frames are rendered
                         via env.reset_to(state, medoid_state) + env.render().
        save_path:       If set, the figure is saved here.
    """
    K = len(perceiver.node_obs)
    colors = _node_colors(K)

    # --- Build per-node obs pools -----------------------------------------------
    if node_state_pool is not None:
        pools = node_state_pool
    else:
        if ls.dataset is None:
            raise RuntimeError("Pass node_state_pool= or attach a dataset to ls.")
        pools = [[] for _ in range(K)]
        for ep in ls.dataset.iterate_episodes():
            for obs in ls._extract_obs(ep.observations):
                pools[perceiver(obs)].append(obs)

    # --- Sample from each pool --------------------------------------------------
    # First column is always the medoid; remaining columns are random samples.
    print("Beginning loop")
    n_extra = max(0, n_samples - 1)
    samples: list[list[np.ndarray]] = []
    for k in range(K):
        print("Node", k)
        pool = pools[k]
        medoid = perceiver.node_obs[k]
        if pool and n_extra > 0:
            idx = np.random.choice(len(pool), min(n_extra, len(pool)), replace=False)
            row = [medoid] + [pool[i] for i in idx]
        else:
            row = [medoid]
        samples.append(row)

    n_cols = max(len(row) for row in samples)
    is_image = ls._is_image_obs or (ls.obs_shape is not None and len(ls.obs_shape) == 3)

    # --- Render frames via env if provided --------------------------------------
    def _render_frame(obs: np.ndarray, node_k: int) -> np.ndarray:
        """Reset env to obs's state (goal = node medoid) and return an RGB frame."""
        state = perceiver.obs_to_state.get(obs.tobytes()) if perceiver.obs_to_state else None
        if state is None:
            state = env.obs_to_state(obs)
        goal_state = perceiver.node_states[node_k]
        env.reset_to(state, goal_state)
        return env.render()

    # --- Draw -------------------------------------------------------------------
    fig, axes = plt.subplots(
        K, n_cols,
        figsize=(n_cols * 1.8, K * 1.8),
        squeeze=False,
    )

    for k in range(K):
        for col in range(n_cols):
            ax = axes[k][col]
            ax.set_xticks([])
            ax.set_yticks([])

            if col < len(samples[k]):
                obs = samples[k][col]

                if env is not None:
                    img = _render_frame(obs, k)
                    ax.imshow(img, interpolation='nearest')
                elif is_image:
                    img = obs
                    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
                        img = np.transpose(img, (1, 2, 0))  # CHW → HWC
                    if img.dtype != np.uint8:
                        lo, hi = img.min(), img.max()
                        img = ((img - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
                    if img.ndim == 3 and img.shape[2] == 1:
                        img = img[..., 0]
                    ax.imshow(img, interpolation='nearest')
                else:
                    vals = obs.flatten()
                    ax.barh(np.arange(len(vals)), vals, color=colors[k], height=0.8)
                    ax.axvline(0, color='black', linewidth=0.5)

                # Highlight the medoid column with a coloured border.
                if col == 0:
                    for spine in ax.spines.values():
                        spine.set_edgecolor(colors[k])
                        spine.set_linewidth(3)
            else:
                ax.axis('off')

        # Row label on the leftmost axes.
        axes[k][0].set_ylabel(
            f'node {k}\n({len(pools[k])} obs)',
            fontsize=8, rotation=0, labelpad=48, va='center',
        )

    render_note = 'env render' if env is not None else ('image obs' if is_image else 'obs features')
    fig.suptitle(
        f'Node sample observations  (K={K}, medoid = col 0, {render_note})',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_demo_nodes(
    ls,
    perceiver,
    episode_idx: int = 0,
    pca: PCA | None = None,
    use_pca: bool = True,
    save_path: str | None = None,
) -> None:
    """Trajectory plot of a single dataset demo in 2-D, colored by perceived node.

    The goal is the demo's final observation (last entry of episode.observations).

    Args:
        ls:          Trained LearningSystem (must have a dataset).
        perceiver:   Output of ls.create_perceiver(K).
        episode_idx: Which episode to plot (0-indexed).
        pca:         PCA fitted by plot_dataset_nodes for a consistent projection.
        use_pca:     If True (default), project with PCA. If False, use dims 0-1.
        save_path:   If set, figure is saved here as well as shown.
    """
    episodes = list(ls.dataset.iterate_episodes())
    ep = episodes[episode_idx]
    obs_arr = ls._extract_obs(ep.observations)  # (T+1, ...)
    obs_list = list(obs_arr)
    goal_obs = obs_list[-1]

    use_obs_space = ls.obs_shape == (2,)
    if use_obs_space:
        coord_label = 'observation space'
    elif use_pca:
        coord_label = 'sym embedding (PCA)'
    else:
        coord_label = 'sym embedding (dims 0-1)'
    K = len(perceiver.node_embeddings)
    title = (
        f'Demo #{episode_idx} trajectory by perceived node\n'
        f'({coord_label}, {len(obs_list)} steps, K={K})'
    )
    _plot_trajectory(
        ls, perceiver,
        obs_list=obs_list,
        goal_obs=goal_obs,
        title=title,
        pca=pca,
        use_pca=use_pca,
        save_path=save_path,
    )
