use std::sync::{Arc, Barrier, mpsc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use crate::solver::mesh::{Mesh, BoundaryType};
use crate::solver::piso::PisoSolver;
use crate::solver::linear_solver::{SparseMatrix, SolverOps};
use nalgebra::{Point2, Vector2};

use std::collections::HashMap;
use crate::solver::fvm::{VectorField, ScalarField, Scheme};

// Communication structures
pub struct Communicator {
    pub rank: usize,
    pub size: usize,
    pub txs: Vec<mpsc::Sender<Message>>,
    pub rxs: Vec<mpsc::Receiver<Message>>,
    pub barrier: Arc<Barrier>,
    pub wait_stats: std::cell::RefCell<HashMap<String, std::time::Duration>>,
}

pub enum Message {
    Sum(f64),
    Halo(usize, Vec<f64>), // tag, data
}

impl Communicator {
    pub fn new(rank: usize, size: usize, txs: Vec<mpsc::Sender<Message>>, rxs: Vec<mpsc::Receiver<Message>>, barrier: Arc<Barrier>) -> Self {
        Self { 
            rank, 
            size, 
            txs, 
            rxs, 
            barrier,
            wait_stats: std::cell::RefCell::new(HashMap::new()),
        }
    }

    pub fn reset_wait_time(&self) {
        self.wait_stats.borrow_mut().clear();
    }

    pub fn record_wait(&self, label: &str, duration: std::time::Duration) {
        let mut stats = self.wait_stats.borrow_mut();
        let entry = stats.entry(label.to_string()).or_insert(std::time::Duration::new(0, 0));
        *entry += duration;
    }

    fn wait_barrier(&self, label: &str) {
        let start = std::time::Instant::now();
        self.barrier.wait();
        let elapsed = start.elapsed();
        self.record_wait(&format!("barrier_{}", label), elapsed);
        if elapsed.as_millis() > 2 {
            println!("Rank {} slow barrier '{}': {:?}", self.rank, label, elapsed);
        }
    }

    pub fn barrier(&self) {
        self.wait_barrier("explicit");
    }

    pub fn all_reduce_sum(&self, val: f64) -> f64 {
        if self.size == 1 { return val; }
        
        // Simple implementation: All send to 0, 0 sums and broadcasts
        if self.rank == 0 {
            let mut sum = val;
            for i in 1..self.size {
                let start = std::time::Instant::now();
                if let Ok(Message::Sum(v)) = self.rxs[i].recv() {
                    let elapsed = start.elapsed();
                    self.record_wait("reduce_recv", elapsed);
                    if elapsed.as_millis() > 2 {
                        println!("Rank {} slow reduce recv from {}: {:?}", self.rank, i, elapsed);
                    }
                    sum += v;
                }
            }
            for i in 1..self.size {
                let _ = self.txs[i].send(Message::Sum(sum));
            }
            self.wait_barrier("reduce_sum");
            sum
        } else {
            let _ = self.txs[0].send(Message::Sum(val));
            let start = std::time::Instant::now();
            let res = if let Ok(Message::Sum(v)) = self.rxs[0].recv() {
                let elapsed = start.elapsed();
                self.record_wait("reduce_recv", elapsed);
                if elapsed.as_millis() > 2 {
                    println!("Rank {} slow reduce recv from 0: {:?}", self.rank, elapsed);
                }
                v
            } else {
                0.0
            };
            self.wait_barrier("reduce_sum");
            res
        }
    }

    pub fn exchange_halo(&self, neighbor_rank: usize, send_data: &[f64]) -> Vec<f64> {
        let _ = self.txs[neighbor_rank].send(Message::Halo(self.rank, send_data.to_vec()));
        
        let start = std::time::Instant::now();
        if let Ok(Message::Halo(_, data)) = self.rxs[neighbor_rank].recv() {
            let elapsed = start.elapsed();
            self.record_wait("halo_recv_single", elapsed);
            if elapsed.as_millis() > 2 {
                println!("Rank {} slow halo recv from {}: {:?}", self.rank, neighbor_rank, elapsed);
            }
            return data;
        }
        Vec::new()
    }
}

pub struct ParallelPisoSolver {
    pub partitions: Vec<Arc<RwLock<SolverPartition>>>,
    pub n_threads: usize,
    workers: Vec<thread::JoinHandle<()>>,
    start_barrier: Arc<Barrier>,
    end_barrier: Arc<Barrier>,
    running: Arc<AtomicBool>,
}

impl ParallelPisoSolver {
    pub fn new(full_mesh: Mesh, n_threads: usize) -> Self {
        let sub_meshes = partition_mesh(full_mesh, n_threads);
        let mut solvers: Vec<PisoSolver> = sub_meshes.into_iter()
            .map(|m| PisoSolver::new(m))
            .collect();
            
        Self::connect_subdomains(&mut solvers);
        let halo_maps = Self::init_parallel_structures(&mut solvers);
        
        let partitions: Vec<_> = solvers.iter()
            .map(|s| Arc::new(RwLock::new(SolverPartition::from_solver(s))))
            .collect();
            
        let mut ps = Self {
            partitions,
            n_threads,
            workers: Vec::new(),
            start_barrier: Arc::new(Barrier::new(n_threads + 1)),
            end_barrier: Arc::new(Barrier::new(n_threads + 1)),
            running: Arc::new(AtomicBool::new(true)),
        };
        ps.spawn_threads(solvers, halo_maps);
        ps
    }

    fn init_parallel_structures(solvers: &mut Vec<PisoSolver>) -> Vec<HaloMap> {
        let n_solvers = solvers.len();
        let mut halo_maps = Vec::new();
        let mut ghost_maps = Vec::new();
        
        for s_idx in 0..n_solvers {
            let mut sends: HashMap<usize, Vec<usize>> = HashMap::new();
            let mut recvs: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            let mut ghost_map: HashMap<usize, usize> = HashMap::new();
            
            let solver = &solvers[s_idx];
            let mesh = &solver.mesh;
            let n_local = mesh.num_cells();
            let mut next_ghost_idx = n_local;
            
            for f_idx in 0..mesh.num_faces() {
                if let Some(BoundaryType::ParallelInterface(target_rank, remote_idx)) = mesh.face_boundary[f_idx] {
                    sends.entry(target_rank).or_default().push(mesh.face_owner[f_idx]);
                    
                    let ghost_idx = next_ghost_idx;
                    next_ghost_idx += 1;
                    
                    recvs.entry(target_rank).or_default().push((ghost_idx, remote_idx));
                    ghost_map.insert(f_idx, ghost_idx);
                }
            }
            
            for list in sends.values_mut() {
                list.sort();
                list.dedup();
            }
            
            let mut final_recvs: HashMap<usize, RecvInfo> = HashMap::new();
            
            for (rank, list) in recvs {
                let mut unique_remotes: Vec<usize> = list.iter().map(|&(_, r)| r).collect();
                unique_remotes.sort();
                unique_remotes.dedup();
                
                let mut map = Vec::new();
                for (ghost_idx, remote_idx) in list {
                    if let Ok(data_idx) = unique_remotes.binary_search(&remote_idx) {
                        map.push((data_idx, ghost_idx));
                    } else {
                        panic!("Should not happen: remote_idx not found in unique list");
                    }
                }
                
                final_recvs.insert(rank, RecvInfo {
                    map,
                    num_unique_cells: unique_remotes.len(),
                });
            }
            
            halo_maps.push(HaloMap {
                sends,
                recvs: final_recvs,
                n_local,
                n_ghosts: next_ghost_idx - n_local,
            });
            ghost_maps.push(ghost_map);
        }
        
        let mut all_ghost_centers = vec![Vec::new(); n_solvers];
        
        for i in 0..n_solvers {
            let mut ghost_centers = vec![Vector2::zeros(); halo_maps[i].n_ghosts];
            
            for (&src_rank, info) in &halo_maps[i].recvs {
                // Get the list of cells sent by src_rank to i
                let sent_indices = &halo_maps[src_rank].sends[&i];
                
                // The data received is in the order of sent_indices.
                // info.map maps (index in sent_indices, ghost_idx).
                
                for &(data_idx, ghost_idx) in &info.map {
                    let remote_cell_idx = sent_indices[data_idx];
                    let cx = solvers[src_rank].mesh.cell_cx[remote_cell_idx];
                    let cy = solvers[src_rank].mesh.cell_cy[remote_cell_idx];
                    
                    let local_ghost_idx = ghost_idx - halo_maps[i].n_local;
                    ghost_centers[local_ghost_idx] = Vector2::new(cx, cy);
                }
            }
            all_ghost_centers[i] = ghost_centers;
        }
        
        for (i, solver) in solvers.iter_mut().enumerate() {
            solver.ghost_map = ghost_maps[i].clone();
            solver.ghost_centers = all_ghost_centers[i].clone();
        }
        
        halo_maps
    }

    fn spawn_threads(&mut self, solvers: Vec<PisoSolver>, halo_maps: Vec<HaloMap>) {
        let n_solvers = solvers.len();
        let comm_barrier = Arc::new(Barrier::new(n_solvers));
        
        let mut all_txs: Vec<Vec<mpsc::Sender<Message>>> = vec![Vec::new(); n_solvers];
        let mut all_rxs: Vec<Vec<Option<mpsc::Receiver<Message>>>> = Vec::with_capacity(n_solvers);
        for _ in 0..n_solvers {
            let mut row = Vec::with_capacity(n_solvers);
            for _ in 0..n_solvers {
                row.push(None);
            }
            all_rxs.push(row);
        }
        
        for i in 0..n_solvers {
            for j in 0..n_solvers {
                let (tx, rx) = mpsc::channel();
                all_txs[i].push(tx);
                all_rxs[i][j] = Some(rx);
            }
        }
        
        let mut thread_rxs: Vec<Vec<mpsc::Receiver<Message>>> = Vec::with_capacity(n_solvers);
        for _ in 0..n_solvers {
            thread_rxs.push(Vec::new());
        }
        for i in 0..n_solvers {
            for j in 0..n_solvers {
                if let Some(rx) = all_rxs[j][i].take() {
                    thread_rxs[i].push(rx);
                }
            }
        }
        
        let mut solvers_iter = solvers.into_iter();
        
        for rank in 0..n_solvers {
            let mut solver = solvers_iter.next().unwrap();
            let start_barrier = self.start_barrier.clone();
            let end_barrier = self.end_barrier.clone();
            let running = self.running.clone();
            let halo_map = halo_maps[rank].clone();
            let my_txs = all_txs[rank].clone();
            let my_rxs = thread_rxs[rank].drain(..).collect();
            let comm_barrier = comm_barrier.clone();
            let partition = self.partitions[rank].clone();
            
            let handle = thread::spawn(move || {
                let comm = Communicator::new(rank, n_solvers, my_txs, my_rxs, comm_barrier);
                
                while running.load(Ordering::Relaxed) {
                    start_barrier.wait();
                    if !running.load(Ordering::Relaxed) { break; }
                    
                    comm.reset_wait_time();
                    
                    {
                        // Sync from partition to solver
                        {
                            let part = partition.read().unwrap();
                            solver.dt = part.dt;
                            solver.density = part.density;
                            solver.viscosity = part.viscosity;
                            solver.scheme = part.scheme;
                            if solver.u.vx.len() == part.u.vx.len() {
                                solver.u = part.u.clone();
                                solver.p = part.p.clone();
                            }
                        }

                        let ops = ParallelOps {
                            comm: &comm,
                            halo_map: &halo_map,
                        };
                        
                        let start_compute = std::time::Instant::now();
                        solver.step_with_ops(&ops);
                        let total_duration = start_compute.elapsed();
                        
                        let stats = comm.wait_stats.borrow();
                        let total_wait: std::time::Duration = stats.values().sum();
                        
                        if total_duration.as_millis() > 10 {
                             println!("Rank {}: Total {:.2}ms, Wait {:.2}ms ({:.1}%)", 
                                rank, 
                                total_duration.as_secs_f64() * 1000.0,
                                total_wait.as_secs_f64() * 1000.0,
                                (total_wait.as_secs_f64() / total_duration.as_secs_f64()) * 100.0
                            );
                            let mut sorted_stats: Vec<_> = stats.iter().collect();
                            sorted_stats.sort_by(|a, b| b.1.cmp(a.1));
                            for (label, duration) in sorted_stats.iter().take(3) {
                                println!("  - {}: {:.2}ms", label, duration.as_secs_f64() * 1000.0);
                            }
                        }

                        // Sync back to partition
                        {
                            let mut part = partition.write().unwrap();
                            part.u = solver.u.clone();
                            part.p = solver.p.clone();
                            part.residuals = solver.residuals.clone();
                            part.time = solver.time;
                        }
                    }
                    
                    end_barrier.wait();
                }
            });
            self.workers.push(handle);
        }
    }

    fn connect_subdomains(solvers: &mut Vec<PisoSolver>) {
        let n_solvers = solvers.len();
        let mut interface_faces = vec![Vec::new(); n_solvers];
        
        // Collect interface faces
        for s_idx in 0..n_solvers {
            let solver = &solvers[s_idx];
            for f_idx in 0..solver.mesh.num_faces() {
                if let Some(BoundaryType::ParallelInterface(_, _)) = solver.mesh.face_boundary[f_idx] {
                    let center = Vector2::new(solver.mesh.face_cx[f_idx], solver.mesh.face_cy[f_idx]);
                    interface_faces[s_idx].push((f_idx, center));
                }
            }
        }
        
        let mut updates = Vec::new();
        
        for s_idx in 0..n_solvers {
            let solver = &solvers[s_idx];
            for &(f_idx, center) in &interface_faces[s_idx] {
                if let Some(BoundaryType::ParallelInterface(target_rank, _)) = solver.mesh.face_boundary[f_idx] {
                    let mut found = false;
                    for &(target_f_idx, target_center) in &interface_faces[target_rank] {
                        if (center - target_center).norm() < 1e-6 {
                            let target_solver = &solvers[target_rank];
                            let remote_cell_idx = target_solver.mesh.face_owner[target_f_idx];
                            updates.push((s_idx, f_idx, target_rank, remote_cell_idx));
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        println!("Rank {} Face {} at {:?} not found in Rank {}", s_idx, f_idx, center, target_rank);
                    }
                }
            }
        }
        
        for (s_idx, f_idx, target_rank, remote_cell_idx) in updates {
            solvers[s_idx].mesh.face_boundary[f_idx] = 
                Some(BoundaryType::ParallelInterface(target_rank, remote_cell_idx));
        }
    }

    pub fn step(&mut self) {
        let start_step = std::time::Instant::now();
        
        self.start_barrier.wait();
        self.end_barrier.wait();
        
        let total_time = start_step.elapsed();
        if total_time.as_millis() > 10 {
             println!("Step time: {:?}", total_time);
        }
    }
}

impl Drop for ParallelPisoSolver {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        // Wake up threads waiting on start_barrier
        self.start_barrier.wait();
        
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}


fn partition_mesh(mesh: Mesh, n_parts: usize) -> Vec<Mesh> {
    if n_parts == 1 {
        return vec![mesh];
    }

    // 1. Sort cells by X coordinate
    let mut cell_indices: Vec<usize> = (0..mesh.num_cells()).collect();
    cell_indices.sort_by(|&a, &b| {
        mesh.cell_cx[a].partial_cmp(&mesh.cell_cx[b]).unwrap()
    });

    // 2. Split indices
    let chunk_size = (cell_indices.len() + n_parts - 1) / n_parts;
    let mut chunks = vec![Vec::new(); n_parts];
    let mut cell_to_part = vec![0; mesh.num_cells()];
    let mut cell_to_local = vec![0; mesh.num_cells()];

    for (i, &cell_idx) in cell_indices.iter().enumerate() {
        let part = i / chunk_size;
        if part < n_parts {
            cell_to_local[cell_idx] = chunks[part].len();
            chunks[part].push(cell_idx);
            cell_to_part[cell_idx] = part;
        }
    }

    // 3. Build sub-meshes
    let mut sub_meshes = Vec::new();

    for part_id in 0..n_parts {
        let mut new_mesh = Mesh::new();
        let my_cells = &chunks[part_id];
        
        // Map global vertex index to local
        let mut global_v_to_local = std::collections::HashMap::new();
        let mut global_face_to_local = std::collections::HashMap::new();
        
        // We iterate over global faces to build local faces
        for f_idx in 0..mesh.num_faces() {
            let owner = mesh.face_owner[f_idx];
            let neighbor = mesh.face_neighbor[f_idx];
            
            let owner_part = cell_to_part[owner];
            let neighbor_part = if let Some(n) = neighbor {
                Some(cell_to_part[n])
            } else {
                None
            };

            if owner_part == part_id {
                // I am owner
                let v0_global = mesh.face_v1[f_idx];
                let v1_global = mesh.face_v2[f_idx];
                
                let v0 = remap_vertex(&mut new_mesh, &mut global_v_to_local, &mesh, v0_global);
                let v1 = remap_vertex(&mut new_mesh, &mut global_v_to_local, &mesh, v1_global);
                
                let local_owner = cell_to_local[owner];
                let local_neighbor = if let Some(n_part) = neighbor_part {
                    if n_part == part_id {
                        Some(cell_to_local[neighbor.unwrap()])
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                let boundary_type = if let Some(n_part) = neighbor_part {
                    if n_part != part_id {
                        Some(BoundaryType::ParallelInterface(n_part, 0))
                    } else {
                        None
                    }
                } else {
                    mesh.face_boundary[f_idx]
                };
                
                let local_idx = new_mesh.num_faces();
                new_mesh.face_v1.push(v0);
                new_mesh.face_v2.push(v1);
                new_mesh.face_owner.push(local_owner);
                new_mesh.face_neighbor.push(local_neighbor);
                new_mesh.face_nx.push(mesh.face_nx[f_idx]);
                new_mesh.face_ny.push(mesh.face_ny[f_idx]);
                new_mesh.face_cx.push(mesh.face_cx[f_idx]);
                new_mesh.face_cy.push(mesh.face_cy[f_idx]);
                new_mesh.face_area.push(mesh.face_area[f_idx]);
                new_mesh.face_boundary.push(boundary_type);
                
                global_face_to_local.insert(f_idx, local_idx);
                
            } else if neighbor_part == Some(part_id) {
                // I am neighbor, so I become owner of this boundary face
                let v0_global = mesh.face_v1[f_idx];
                let v1_global = mesh.face_v2[f_idx];
                
                let v0 = remap_vertex(&mut new_mesh, &mut global_v_to_local, &mesh, v0_global);
                let v1 = remap_vertex(&mut new_mesh, &mut global_v_to_local, &mesh, v1_global);
                
                let local_owner = cell_to_local[neighbor.unwrap()];
                
                // Flip normal and vertices
                let local_idx = new_mesh.num_faces();
                new_mesh.face_v1.push(v1);
                new_mesh.face_v2.push(v0);
                new_mesh.face_owner.push(local_owner);
                new_mesh.face_neighbor.push(None);
                new_mesh.face_nx.push(-mesh.face_nx[f_idx]);
                new_mesh.face_ny.push(-mesh.face_ny[f_idx]);
                new_mesh.face_cx.push(mesh.face_cx[f_idx]);
                new_mesh.face_cy.push(mesh.face_cy[f_idx]);
                new_mesh.face_area.push(mesh.face_area[f_idx]);
                new_mesh.face_boundary.push(Some(BoundaryType::ParallelInterface(owner_part, 0)));
                
                global_face_to_local.insert(f_idx, local_idx);
            }
        }
        
        // Construct Cells
        for &global_cell_idx in my_cells {
            new_mesh.cell_cx.push(mesh.cell_cx[global_cell_idx]);
            new_mesh.cell_cy.push(mesh.cell_cy[global_cell_idx]);
            new_mesh.cell_vol.push(mesh.cell_vol[global_cell_idx]);
            
            // Faces
            let start_f = mesh.cell_face_offsets[global_cell_idx];
            let end_f = mesh.cell_face_offsets[global_cell_idx+1];
            new_mesh.cell_face_offsets.push(new_mesh.cell_faces.len());
            for k in start_f..end_f {
                let global_f_idx = mesh.cell_faces[k];
                if let Some(&local_f_idx) = global_face_to_local.get(&global_f_idx) {
                    new_mesh.cell_faces.push(local_f_idx);
                }
            }
            
            // Vertices
            let start_v = mesh.cell_vertex_offsets[global_cell_idx];
            let end_v = mesh.cell_vertex_offsets[global_cell_idx+1];
            new_mesh.cell_vertex_offsets.push(new_mesh.cell_vertices.len());
            for k in start_v..end_v {
                let global_v_idx = mesh.cell_vertices[k];
                if let Some(&local_v_idx) = global_v_to_local.get(&global_v_idx) {
                    new_mesh.cell_vertices.push(local_v_idx);
                }
            }
        }
        // Push final offsets
        new_mesh.cell_face_offsets.push(new_mesh.cell_faces.len());
        new_mesh.cell_vertex_offsets.push(new_mesh.cell_vertices.len());
        
        sub_meshes.push(new_mesh);
    }

    sub_meshes
}

fn remap_vertex(
    new_mesh: &mut Mesh, 
    map: &mut std::collections::HashMap<usize, usize>, 
    old_mesh: &Mesh, 
    global_idx: usize
) -> usize {
    *map.entry(global_idx).or_insert_with(|| {
        let idx = new_mesh.num_vertices();
        new_mesh.vx.push(old_mesh.vx[global_idx]);
        new_mesh.vy.push(old_mesh.vy[global_idx]);
        new_mesh.v_fixed.push(old_mesh.v_fixed[global_idx]);
        idx
    })
}

#[derive(Clone)]
struct RecvInfo {
    map: Vec<(usize, usize)>, // (data_idx, ghost_idx)
    num_unique_cells: usize,
}

#[derive(Clone)]
struct HaloMap {
    sends: HashMap<usize, Vec<usize>>, // rank -> list of local cell indices
    recvs: HashMap<usize, RecvInfo>, // rank -> RecvInfo
    n_local: usize,
    n_ghosts: usize,
}

struct ParallelOps<'a> {
    comm: &'a Communicator,
    halo_map: &'a HaloMap,
}

impl<'a> SolverOps for ParallelOps<'a> {
    fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        let local_dot = crate::solver::linear_solver::dot(a, b);
        self.comm.all_reduce_sum(local_dot)
    }

    fn exchange_halo(&self, data: &[f64]) -> Vec<f64> {
        let mut ghosts = vec![0.0; self.halo_map.n_ghosts];
        
        for (&rank, send_indices) in &self.halo_map.sends {
            let send_data: Vec<f64> = send_indices.iter().map(|&idx| data[idx]).collect();
            let _ = self.comm.txs[rank].send(Message::Halo(self.comm.rank, send_data));
        }
        
        for (&rank, info) in &self.halo_map.recvs {
            let start = std::time::Instant::now();
            if let Ok(Message::Halo(_, recv_data)) = self.comm.rxs[rank].recv() {
                let elapsed = start.elapsed();
                self.comm.record_wait("halo_recv", elapsed);
                if elapsed.as_millis() > 2 {
                    println!("Rank {} slow halo recv from {}: {:?}", self.comm.rank, rank, elapsed);
                }
                if recv_data.len() != info.num_unique_cells {
                    panic!("Halo exchange size mismatch with rank {}. Expected {}, got {}", rank, info.num_unique_cells, recv_data.len());
                }
                for &(data_idx, ghost_idx) in &info.map {
                    let local_ghost_idx = ghost_idx - self.halo_map.n_local;
                    ghosts[local_ghost_idx] = recv_data[data_idx];
                }
            }
        }
        ghosts
    }
    
    fn mat_vec_mul(&self, matrix: &SparseMatrix, x: &[f64], y: &mut [f64]) {
        let ghosts = self.exchange_halo(x);
        let mut x_extended = Vec::with_capacity(self.halo_map.n_local + self.halo_map.n_ghosts);
        x_extended.extend_from_slice(x);
        x_extended.extend_from_slice(&ghosts);
        
        matrix.mat_vec_mul(&x_extended, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::mesh::{ChannelWithObstacle, generate_cut_cell_mesh};
    use nalgebra::{Point2, Vector2};

    #[test]
    fn test_parallel_vs_serial() {
        use crate::solver::mesh::{RectangularChannel, generate_cut_cell_mesh};
        
        let geo = RectangularChannel {
            length: 2.0,
            height: 1.0,
        };
        let mesh = generate_cut_cell_mesh(&geo, 0.1, 0.1, Vector2::new(2.0, 1.0));
        
        // Serial
        let mut serial_solver = PisoSolver::new(mesh.clone());
        for _ in 0..5 {
            serial_solver.step();
        }
        
        // Parallel
        let mut parallel_solver = ParallelPisoSolver::new(mesh.clone(), 4);
        for _ in 0..5 {
            parallel_solver.step();
        }
        
        // Compare
        for partition in &parallel_solver.partitions {
            let solver = partition.read().unwrap();
            for i in 0..solver.mesh.num_cells() {
                let center = Point2::new(solver.mesh.cell_cx[i], solver.mesh.cell_cy[i]);
                // Find corresponding cell in serial mesh
                if let Some(serial_idx) = serial_solver.mesh.get_cell_at_pos(center) {
                    let p_serial = serial_solver.p.values[serial_idx];
                    let p_parallel = solver.p.values[i];
                    
                    if (p_serial - p_parallel).abs() > 1e-3 {
                        println!("Pressure mismatch at {:?}: serial {}, parallel {}", center, p_serial, p_parallel);
                    }
                }
            }
        }
    }

    #[test]
    fn test_parallel_vs_serial_backward_step() {
        use crate::solver::mesh::{BackwardsStep, Mesh, BoundaryType};
        use std::collections::{HashSet, VecDeque};

        fn get_connected_cells(mesh: &Mesh) -> HashSet<usize> {
            let mut visited = vec![false; mesh.num_cells()];
            let mut queue = VecDeque::new();
            
            // Find outlet cells
            for i in 0..mesh.num_cells() {
                let start = mesh.cell_face_offsets[i];
                let end = mesh.cell_face_offsets[i+1];
                for k in start..end {
                    let face_idx = mesh.cell_faces[k];
                    if let Some(bt) = mesh.face_boundary[face_idx] {
                        if matches!(bt, BoundaryType::Outlet) {
                            visited[i] = true;
                            queue.push_back(i);
                            break;
                        }
                    }
                }
            }
            
            while let Some(i) = queue.pop_front() {
                let start = mesh.cell_face_offsets[i];
                let end = mesh.cell_face_offsets[i+1];
                for k in start..end {
                    let face_idx = mesh.cell_faces[k];
                    let owner = mesh.face_owner[face_idx];
                    let neighbor = mesh.face_neighbor[face_idx];
                    let n_idx = if owner == i { neighbor } else { Some(owner) };
                    
                    if let Some(n) = n_idx {
                        if !visited[n] {
                            visited[n] = true;
                            queue.push_back(n);
                        }
                    }
                }
            }
            
            visited.iter().enumerate().filter_map(|(i, &v)| if v { Some(i) } else { None }).collect()
        }

        let length = 3.5;
        let domain_size = Vector2::new(length, 1.0);
        let geo = BackwardsStep {
            length,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 0.5,
        };
        let min_cell_size = 0.025;
        let max_cell_size = 0.025;
        let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
        mesh.smooth(0.3, 50);
        
        let dt = 0.01;
        let density = 1000.0;
        let viscosity = 0.001;
        
        // Serial
        let mut serial_solver = PisoSolver::new(mesh.clone());
        serial_solver.dt = dt;
        serial_solver.density = density;
        serial_solver.viscosity = viscosity;
        // BCs
        for i in 0..serial_solver.mesh.num_cells() {
            let cx = serial_solver.mesh.cell_cx[i];
            let cy = serial_solver.mesh.cell_cy[i];
            if cx < max_cell_size {
                if cy > 0.5 {
                    serial_solver.u.vx[i] = 1.0;
                    serial_solver.u.vy[i] = 0.0;
                }
            }
        }
        
        for _ in 0..10 {
            serial_solver.step();
        }
        
        // Parallel
        let mut parallel_solver = ParallelPisoSolver::new(mesh.clone(), 4);
        for partition in &parallel_solver.partitions {
            let mut solver = partition.write().unwrap();
            solver.dt = dt;
            solver.density = density;
            solver.viscosity = viscosity;
            // BCs
            let n_cells = solver.mesh.num_cells();
            for i in 0..n_cells {
                let cx = solver.mesh.cell_cx[i];
                let cy = solver.mesh.cell_cy[i];
                if cx < max_cell_size {
                    if cy > 0.5 {
                        solver.u.vx[i] = 1.0;
                        solver.u.vy[i] = 0.0;
                    }
                }
            }
        }
        
        for _ in 0..10 {
            parallel_solver.step();
        }
        
        // Identify connected cells in serial mesh
        let connected_cells = get_connected_cells(&serial_solver.mesh);
        println!("Connected cells: {}/{}", connected_cells.len(), serial_solver.mesh.num_cells());

        // Compare
        let mut max_u_diff = 0.0;
        for partition in &parallel_solver.partitions {
            let solver = partition.read().unwrap();
            for i in 0..solver.mesh.num_cells() {
                let center = Point2::new(solver.mesh.cell_cx[i], solver.mesh.cell_cy[i]);
                if let Some(serial_idx) = serial_solver.mesh.get_cell_at_pos(center) {
                    // Only compare if connected
                    if !connected_cells.contains(&serial_idx) {
                        continue;
                    }

                    let u_serial = Vector2::new(serial_solver.u.vx[serial_idx], serial_solver.u.vy[serial_idx]);
                    let u_parallel = Vector2::new(solver.u.vx[i], solver.u.vy[i]);
                    
                    let u_diff = (u_serial - u_parallel).norm();
                    if u_diff > max_u_diff {
                        max_u_diff = u_diff;
                    }

                    assert!((u_serial - u_parallel).norm() < 0.1, "Velocity mismatch at {:?}: serial {}, parallel {}", center, u_serial, u_parallel);
                }
            }
        }
        println!("Max Velocity Mismatch: {}", max_u_diff);
    }

    #[test]
    #[ignore]
    fn test_profiling_large_mesh() {
        use crate::solver::mesh::{RectangularChannel, generate_cut_cell_mesh};
        
        println!("Generating large mesh...");
        let geo = RectangularChannel {
            length: 2.0,
            height: 1.0,
        };
        // 200x100 = 20,000 cells
        let mesh = generate_cut_cell_mesh(&geo, 0.01, 0.01, Vector2::new(2.0, 1.0));
        println!("Mesh generated: {} cells", mesh.num_cells());
        
        let mut solver = ParallelPisoSolver::new(mesh, 4);
        
        println!("Starting 5 steps...");
        let start = std::time::Instant::now();
        for i in 0..5 {
            solver.step();
            println!("Step {} done", i);
        }
        let elapsed = start.elapsed();
        println!("Total time for 5 steps: {:?}", elapsed);
        println!("Average time per step: {:?}", elapsed / 5);
    }

    #[test]
    #[ignore]
    fn test_profiling_large_backwards_step() {
        use crate::solver::mesh::{BackwardsStep, Mesh, BoundaryType};
        
        println!("Generating large backwards step mesh...");
        let length = 5.0;
        let domain_size = Vector2::new(length, 1.0);
        let geo = BackwardsStep {
            length,
            height_inlet: 0.5,
            height_outlet: 1.0,
            step_x: 1.0,
        };
        // 5.0 * 1.0 - 1.0 * 0.5 = 4.5 m^2
        // 0.003 cell size -> 9e-6 m^2 per cell -> 500,000 cells
        let min_cell_size = 0.003;
        let max_cell_size = 0.003;
        let mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
        println!("Mesh generated: {} cells", mesh.num_cells());
        
        let n_threads = 4;
        let mut solver = ParallelPisoSolver::new(mesh, n_threads);
        
        // Setup BCs
        for partition in &solver.partitions {
            let mut s = partition.write().unwrap();
            s.dt = 0.001;
            s.density = 1.225;
            s.viscosity = 1.81e-5;
            
            let n_cells = s.mesh.num_cells();
            for i in 0..n_cells {
                let cx = s.mesh.cell_cx[i];
                let cy = s.mesh.cell_cy[i];
                if cx < max_cell_size {
                    if cy > 0.5 {
                        s.u.vx[i] = 1.0;
                        s.u.vy[i] = 0.0;
                    }
                }
            }
        }
        
        println!("Starting 5 steps...");
        let start = std::time::Instant::now();
        for i in 0..5 {
            solver.step();
        }
        let elapsed = start.elapsed();
        println!("Total time for 5 steps: {:?}", elapsed);
        println!("Average time per step: {:?}", elapsed / 5);
    }

    #[test]
    fn test_parallel_vs_serial_channel_with_obstacle() {
        use crate::solver::mesh::{ChannelWithObstacle, generate_cut_cell_mesh};
        use std::collections::HashSet;

        fn get_connected_cells(mesh: &Mesh) -> HashSet<usize> {
            let mut connected = HashSet::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0);
            connected.insert(0);
            while let Some(cell_idx) = queue.pop_front() {
                let start_face = mesh.cell_face_offsets[cell_idx];
                let end_face = mesh.cell_face_offsets[cell_idx + 1];
                for i in start_face..end_face {
                    let face_idx = mesh.cell_faces[i];
                    let owner = mesh.face_owner[face_idx];
                    let neighbor = mesh.face_neighbor[face_idx];
                    let next_cell = if owner == cell_idx { neighbor } else { Some(owner) };
                    if let Some(n_idx) = next_cell {
                        if !connected.contains(&n_idx) {
                            connected.insert(n_idx);
                            queue.push_back(n_idx);
                        }
                    }
                }
            }
            connected
        }

        println!("Generating channel with obstacle mesh...");
        let geo = ChannelWithObstacle {
            length: 4.0,
            height: 1.0,
            obstacle_center: Point2::new(1.0, 0.5),
            obstacle_radius: 0.1,
        };
        let domain_size = Vector2::new(2.0, 1.0);
        let cell_size = 0.025; 
        let mesh = generate_cut_cell_mesh(&geo, cell_size, cell_size, domain_size);
        println!("Mesh generated: {} cells", mesh.num_cells());

        // Serial Solver
        let mut serial_solver = PisoSolver::new(mesh.clone());
        serial_solver.dt = 0.001;
        serial_solver.density = 1.0;
        serial_solver.viscosity = 0.01;

        // Set Serial BCs (Inlet at x=0)
        for i in 0..serial_solver.mesh.num_cells() {
            let cx = serial_solver.mesh.cell_cx[i];
            if cx < cell_size {
                serial_solver.u.vx[i] = 1.0;
                serial_solver.u.vy[i] = 0.0;
            }
        }

        // Parallel Solver
        let n_threads = 4;
        let mut parallel_solver = ParallelPisoSolver::new(mesh.clone(), n_threads);
        
        // Set Parallel BCs
        for partition in &parallel_solver.partitions {
            let mut s = partition.write().unwrap();
            s.dt = 0.001;
            s.density = 1.0;
            s.viscosity = 0.01;
            
            let n_cells = s.mesh.num_cells();
            for i in 0..n_cells {
                let cx = s.mesh.cell_cx[i];
                if cx < cell_size {
                    s.u.vx[i] = 1.0;
                    s.u.vy[i] = 0.0;
                }
            }
        }

        let connected_cells = get_connected_cells(&serial_solver.mesh);
        println!("Connected cells: {}/{}", connected_cells.len(), serial_solver.mesh.num_cells());

        println!("Starting comparison loop...");
        for step in 0..40 {
            serial_solver.step();
            parallel_solver.step();

            let mut max_u_diff = 0.0;
            let mut max_p_diff = 0.0;
            let mut max_u_diff_loc = Point2::origin();
            let mut max_p_diff_loc = Point2::origin();
            
            // Calculate pressure offset (mean pressure difference)
            let mut p_serial_sum = 0.0;
            let mut p_parallel_sum = 0.0;
            let mut count = 0;

             for partition in &parallel_solver.partitions {
                let solver = partition.read().unwrap();
                for i in 0..solver.mesh.num_cells() {
                    let center = Point2::new(solver.mesh.cell_cx[i], solver.mesh.cell_cy[i]);
                    if let Some(serial_idx) = serial_solver.mesh.get_cell_at_pos(center) {
                        if !connected_cells.contains(&serial_idx) { continue; }
                        p_serial_sum += serial_solver.p.values[serial_idx];
                        p_parallel_sum += solver.p.values[i];
                        count += 1;
                    }
                }
            }
            let p_offset = if count > 0 { (p_serial_sum - p_parallel_sum) / count as f64 } else { 0.0 };


            for partition in &parallel_solver.partitions {
                let solver = partition.read().unwrap();
                for i in 0..solver.mesh.num_cells() {
                    let center = Point2::new(solver.mesh.cell_cx[i], solver.mesh.cell_cy[i]);
                    if let Some(serial_idx) = serial_solver.mesh.get_cell_at_pos(center) {
                        if !connected_cells.contains(&serial_idx) { continue; }

                        let u_serial = Vector2::new(serial_solver.u.vx[serial_idx], serial_solver.u.vy[serial_idx]);
                        let u_parallel = Vector2::new(solver.u.vx[i], solver.u.vy[i]);
                        let u_diff = (u_serial - u_parallel).norm();

                        let p_serial = serial_solver.p.values[serial_idx];
                        let p_parallel = solver.p.values[i];
                        let p_diff = (p_serial - (p_parallel + p_offset)).abs();

                        if u_diff > max_u_diff {
                            max_u_diff = u_diff;
                            max_u_diff_loc = center;
                        }
                        if p_diff > max_p_diff {
                            max_p_diff = p_diff;
                            max_p_diff_loc = center;
                        }
                    }
                }
            }
            
            println!("Step {}: Max U diff = {:.6} at {:?}, Max P diff (adjusted) = {:.6} at {:?}", 
                step, max_u_diff, max_u_diff_loc, max_p_diff, max_p_diff_loc);
            
            // Fail if discrepancies are too large
             if max_u_diff > 0.1 {
                println!("Large velocity discrepancy detected!");
            }
        }
    }
}

pub struct SolverPartition {
    pub mesh: Mesh,
    pub u: VectorField,
    pub p: ScalarField,
    pub residuals: Vec<(String, f64)>,
    pub time: f64,
    pub dt: f64,
    pub density: f64,
    pub viscosity: f64,
    pub scheme: Scheme,
}

impl SolverPartition {
    fn from_solver(solver: &PisoSolver) -> Self {
        Self {
            mesh: solver.mesh.clone(),
            u: solver.u.clone(),
            p: solver.p.clone(),
            residuals: solver.residuals.clone(),
            time: solver.time,
            dt: solver.dt,
            density: solver.density,
            viscosity: solver.viscosity,
            scheme: solver.scheme,
        }
    }
}

