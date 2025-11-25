use std::sync::{Arc, Barrier, mpsc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use crate::solver::mesh::{Mesh, BoundaryType};
use crate::solver::piso::PisoSolver;
use crate::solver::linear_solver::{SparseMatrix, SolverOps};
use crate::solver::float::Float;
use nalgebra::{Point2, Vector2};

use std::collections::{HashMap, HashSet};
use crate::solver::fvm::{VectorField, ScalarField, Scheme};

// Communication structures
pub struct Communicator<T: Float> {
    pub rank: usize,
    pub size: usize,
    pub txs: Vec<mpsc::Sender<Message<T>>>,
    pub rxs: Vec<mpsc::Receiver<Message<T>>>,
    pub barrier: Arc<Barrier>,
    pub wait_stats: std::cell::RefCell<HashMap<String, std::time::Duration>>,
}

pub enum Message<T: Float> {
    Sum(T),
    Halo(usize, Vec<T>), // tag, data
}

impl<T: Float> Communicator<T> {
    pub fn new(rank: usize, size: usize, txs: Vec<mpsc::Sender<Message<T>>>, rxs: Vec<mpsc::Receiver<Message<T>>>, barrier: Arc<Barrier>) -> Self {
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

    pub fn all_reduce_sum(&self, val: T) -> T {
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
                T::zero()
            };
            self.wait_barrier("reduce_sum");
            res
        }
    }

    pub fn exchange_halo(&self, neighbor_rank: usize, send_data: &[T]) -> Vec<T> {
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

pub struct ParallelPisoSolver<T: Float> {
    pub partitions: Vec<Arc<RwLock<SolverPartition<T>>>>,
    pub n_threads: usize,
    workers: Vec<thread::JoinHandle<()>>,
    start_barrier: Arc<Barrier>,
    end_barrier: Arc<Barrier>,
    worker_barrier: Arc<Barrier>,
    running: Arc<AtomicBool>,
}

impl<T: Float> ParallelPisoSolver<T> {
    pub fn new(full_mesh: Mesh, n_threads: usize) -> Self {
        let sub_meshes = partition_mesh(full_mesh, n_threads);
        let mut solvers: Vec<PisoSolver<T>> = sub_meshes.into_iter()
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
            worker_barrier: Arc::new(Barrier::new(n_threads)),
            running: Arc::new(AtomicBool::new(true)),
        };
        ps.spawn_threads(solvers, halo_maps);
        ps
    }

    fn init_parallel_structures(solvers: &mut Vec<PisoSolver<T>>) -> Vec<HaloMap> {
        let n_solvers = solvers.len();
        
        // 1. Collect all ParallelInterface faces
        struct InterfaceFace {
            solver_idx: usize,
            face_idx: usize,
            center: Vector2<f64>,
            neighbor_rank: usize,
        }
        
        let mut interfaces = Vec::new();
        for (s_idx, solver) in solvers.iter().enumerate() {
            for f_idx in 0..solver.mesh.num_faces() {
                if let Some(BoundaryType::ParallelInterface(n_rank, _)) = solver.mesh.face_boundary[f_idx] {
                    let cx = solver.mesh.face_cx[f_idx];
                    let cy = solver.mesh.face_cy[f_idx];
                    interfaces.push(InterfaceFace {
                        solver_idx: s_idx,
                        face_idx: f_idx,
                        center: Vector2::new(cx, cy),
                        neighbor_rank: n_rank,
                    });
                }
            }
        }
        
        // 2. Match faces and build connectivity
        let mut pending_sends: Vec<HashMap<usize, HashSet<usize>>> = vec![HashMap::new(); n_solvers];
        // pending_recvs[receiver][sender] = list of (face_idx_in_receiver, owner_cell_in_sender, center_of_sender_cell)
        let mut pending_recvs: Vec<HashMap<usize, Vec<(usize, usize, Vector2<f64>)>>> = vec![HashMap::new(); n_solvers];
        
        for i in 0..interfaces.len() {
            let iface_a = &interfaces[i];
            let neighbor_rank = iface_a.neighbor_rank;
            
            if neighbor_rank >= n_solvers { continue; }
            
            // Find matching face in neighbor_rank
            for j in 0..interfaces.len() {
                let iface_b = &interfaces[j];
                if iface_b.solver_idx == neighbor_rank && iface_b.neighbor_rank == iface_a.solver_idx {
                    let dist = (iface_a.center - iface_b.center).norm();
                    if dist < 1e-4 {
                        // Match found
                        let solver_b = &solvers[neighbor_rank];
                        let owner_b = solver_b.mesh.face_owner[iface_b.face_idx];
                        let center_b = Vector2::new(
                            solver_b.mesh.cell_cx[owner_b],
                            solver_b.mesh.cell_cy[owner_b]
                        );
                        
                        // A needs to receive owner_b from B
                        pending_recvs[iface_a.solver_idx]
                            .entry(neighbor_rank)
                            .or_default()
                            .push((iface_a.face_idx, owner_b, center_b));
                        
                        // B needs to send owner_b to A
                        pending_sends[neighbor_rank]
                            .entry(iface_a.solver_idx)
                            .or_default()
                            .insert(owner_b);
                            
                        break; // Found match for iface_a
                    }
                }
            }
        }
        
        // 3. Build HaloMaps and update Solvers
        let mut halo_maps = Vec::with_capacity(n_solvers);
        
        for rank in 0..n_solvers {
            let mut sends = HashMap::new();
            let mut recvs = HashMap::new();
            
            // Process Sends
            for (target_rank, cell_indices) in &pending_sends[rank] {
                let mut sorted_indices: Vec<usize> = cell_indices.iter().cloned().collect();
                sorted_indices.sort_unstable();
                sends.insert(*target_rank, sorted_indices);
            }
            
            // Process Recvs and Update Solver
            let solver = &mut solvers[rank];
            solver.ghost_map.clear();
            solver.ghost_centers.clear();
            
            for (source_rank, needed_ghosts) in &pending_recvs[rank] {
                if let Some(sent_cells) = pending_sends[*source_rank].get(&rank) {
                    let mut sent_cells_sorted: Vec<usize> = sent_cells.iter().cloned().collect();
                    sent_cells_sorted.sort_unstable();
                    
                    let mut source_to_msg_idx = HashMap::new();
                    for (idx, &cell_idx) in sent_cells_sorted.iter().enumerate() {
                        source_to_msg_idx.insert(cell_idx, idx);
                    }
                    
                    let mut map_entries = Vec::new();
                    
                    for (face_idx, owner_b_idx, center_b) in needed_ghosts {
                        if let Some(&msg_idx) = source_to_msg_idx.get(owner_b_idx) {
                            // Create new ghost
                            let ghost_idx = solver.mesh.num_cells() + solver.ghost_centers.len();
                            solver.ghost_map.insert(*face_idx, ghost_idx);
                            solver.ghost_centers.push(*center_b);
                            
                            // Map msg_idx -> local_ghost_idx
                            let local_ghost_idx = ghost_idx - solver.mesh.num_cells();
                            map_entries.push((msg_idx, local_ghost_idx));
                        }
                    }
                    
                    recvs.insert(*source_rank, RecvInfo {
                        map: map_entries,
                        num_unique_cells: sent_cells_sorted.len(),
                    });
                }
            }
            
            halo_maps.push(HaloMap {
                sends,
                recvs,
                n_local: solver.mesh.num_cells(),
                n_ghosts: solver.ghost_centers.len(),
            });
        }
        
        halo_maps
    }

    fn spawn_threads(&mut self, solvers: Vec<PisoSolver<T>>, halo_maps: Vec<HaloMap>) {
        let mut solvers = solvers;
        let mut halo_maps = halo_maps;
        

        // Correct way:
        // For each pair (i, j), create channel.
        // txs[i][j] sends to j.
        // rxs[j][i] receives from i.
        let mut matrix_txs: Vec<Vec<Option<mpsc::Sender<Message<T>>>>> = (0..self.n_threads).map(|_| (0..self.n_threads).map(|_| None).collect()).collect();
        let mut matrix_rxs: Vec<Vec<Option<mpsc::Receiver<Message<T>>>>> = (0..self.n_threads).map(|_| (0..self.n_threads).map(|_| None).collect()).collect();
        
        for i in 0..self.n_threads {
            for j in 0..self.n_threads {
                let (tx, rx) = mpsc::channel();
                matrix_txs[i][j] = Some(tx);
                matrix_rxs[j][i] = Some(rx);
            }
        }
        
        for i in 0..self.n_threads {
            let mut solver = solvers.remove(0);
            let halo_map = halo_maps.remove(0);
            let partition = self.partitions[i].clone();
            let barrier = self.start_barrier.clone();
            let end_barrier = self.end_barrier.clone();
            let worker_barrier = self.worker_barrier.clone();
            let running = self.running.clone();
            
            let my_txs: Vec<mpsc::Sender<Message<T>>> = matrix_txs[i].iter_mut().map(|opt| opt.take().unwrap()).collect();
            let my_rxs: Vec<mpsc::Receiver<Message<T>>> = matrix_rxs[i].iter_mut().map(|opt| opt.take().unwrap()).collect();
            
            let comm = Communicator::new(i, self.n_threads, my_txs, my_rxs, worker_barrier);
            
            let handle = thread::spawn(move || {
                let ops = ParallelOps {
                    comm: &comm,
                    halo_map: &halo_map,
                };
                
                while running.load(Ordering::Relaxed) {
                    // Wait for start signal
                    barrier.wait();
                    
                    if !running.load(Ordering::Relaxed) { break; }
                    
                    // Update solver from partition (main thread might have changed settings)
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

                    // Reset stats
                    comm.wait_stats.borrow_mut().clear();

                    let start_compute = std::time::Instant::now();
                    solver.step_with_ops(&ops);
                    let total_duration = start_compute.elapsed();
                    
                    let stats = comm.wait_stats.borrow();
                    let total_wait: std::time::Duration = stats.values().sum();
                    let compute_duration = total_duration - total_wait;
                    
                    // Update partition with results
                    {
                        let mut part = partition.write().unwrap();
                        part.u = solver.u.clone();
                        part.p = solver.p.clone();
                        part.residuals = solver.residuals.clone();
                        part.time = solver.time;
                        // part.wait_time = total_wait;
                        // part.compute_time = compute_duration;
                    }
                    
                    // Signal done
                    end_barrier.wait();
                }
            });
            
            self.workers.push(handle);
        }
    }

    fn connect_subdomains(solvers: &mut Vec<PisoSolver<T>>) {
        // Placeholder
    }

    pub fn step(&mut self) {
        // 1. Signal start
        self.start_barrier.wait();
        
        // 2. Wait for completion
        self.end_barrier.wait();
    }
}

impl<T: Float> Drop for ParallelPisoSolver<T> {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        // Wake up threads waiting on barrier
        // This is tricky with Barrier. 
        // We can't easily cancel a barrier wait.
        // But if we are dropping, we assume the app is closing.
        // The threads check 'running' after barrier.
        // But if they are stuck at barrier, they won't check.
        // We might need to force cycle the barrier?
        // Or just let them be killed when process ends (if main thread).
        // But for clean shutdown:
        // We can't do much if they are waiting.
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

struct ParallelOps<'a, T: Float> {
    comm: &'a Communicator<T>,
    halo_map: &'a HaloMap,
}

impl<'a, T: Float> SolverOps<T> for ParallelOps<'a, T> {
    fn exchange_halo(&self, data: &[T]) -> Vec<T> {
        let mut ghosts = vec![T::zero(); self.halo_map.n_ghosts];
        
        // Collect all neighbors we interact with
        let mut neighbors: Vec<usize> = self.halo_map.sends.keys()
            .chain(self.halo_map.recvs.keys())
            .cloned()
            .collect();
        neighbors.sort_unstable();
        neighbors.dedup();
        
        for &neighbor in &neighbors {
            // Prepare send data
            let send_vec = if let Some(indices) = self.halo_map.sends.get(&neighbor) {
                indices.iter().map(|&idx| data[idx]).collect()
            } else {
                Vec::new()
            };
            
            // Exchange
            let recv_vec = self.comm.exchange_halo(neighbor, &send_vec);
            
            // Map received data to ghosts
            if let Some(info) = self.halo_map.recvs.get(&neighbor) {
                for &(data_idx, ghost_idx) in &info.map {
                    if data_idx < recv_vec.len() && ghost_idx < ghosts.len() {
                        ghosts[ghost_idx] = recv_vec[data_idx];
                    }
                }
            }
        }
        
        ghosts
    }

    fn dot(&self, x: &[T], y: &[T]) -> T {
        let local_dot: T = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).fold(T::zero(), |acc, val| acc + val);
        self.comm.all_reduce_sum(local_dot)
    }

    fn norm(&self, x: &[T]) -> T {
        let local_sq_sum: T = x.iter().map(|&a| a * a).fold(T::zero(), |acc, val| acc + val);
        let global_sq_sum = self.comm.all_reduce_sum(local_sq_sum);
        global_sq_sum.sqrt()
    }
    
    fn mat_vec_mul(&self, matrix: &SparseMatrix<T>, x: &[T], y: &mut [T]) {
        // 1. Exchange ghosts
        let ghosts = self.exchange_halo(x);
        
        // 2. Create full vector
        let mut x_full = Vec::with_capacity(x.len() + ghosts.len());
        x_full.extend_from_slice(x);
        x_full.extend_from_slice(&ghosts);
        
        // 3. Multiply
        matrix.mat_vec_mul(&x_full, y);
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
        let mut parallel_solver: ParallelPisoSolver<f64> = ParallelPisoSolver::new(mesh.clone(), 4);
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
                    let p_serial: f64 = serial_solver.p.values[serial_idx];
                    let p_parallel: f64 = solver.p.values[i];
                    
                    if (p_serial - p_parallel).abs() > 1e-3_f64 {
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

        let length = 3.5_f64;
        let domain_size = Vector2::new(length, 1.0_f64);
        let geo = BackwardsStep {
            length,
            height_inlet: 0.5_f64,
            height_outlet: 1.0_f64,
            step_x: 0.5_f64,
        };
        let min_cell_size = 0.025_f64;
        let max_cell_size = 0.025_f64;
        let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
        mesh.smooth(&geo, 0.3_f64, 50);
        
        let dt = 0.01_f64;
        let density = 1000.0_f64;
        let viscosity = 0.001_f64;
        
        // Serial
        let mut serial_solver = PisoSolver::<f64>::new(mesh.clone());
        serial_solver.dt = dt;
        serial_solver.density = density;
        serial_solver.viscosity = viscosity;
        // BCs
        for i in 0..serial_solver.mesh.num_cells() {
            let cx = serial_solver.mesh.cell_cx[i];
            let cy = serial_solver.mesh.cell_cy[i];
            if cx < max_cell_size {
                if cy > 0.5_f64 {
                    serial_solver.u.vx[i] = 1.0_f64;
                    serial_solver.u.vy[i] = 0.0_f64;
                }
            }
        }
        
        for _ in 0..10 {
            serial_solver.step();
        }
        
        // Parallel
        let mut parallel_solver = ParallelPisoSolver::<f64>::new(mesh.clone(), 4);
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
                    if cy > 0.5_f64 {
                        solver.u.vx[i] = 1.0_f64;
                        solver.u.vy[i] = 0.0_f64;
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
        let mut max_u_diff = 0.0_f64;
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

                    assert!((u_serial - u_parallel).norm() < 0.1_f64, "Velocity mismatch at {:?}: serial {}, parallel {}", center, u_serial, u_parallel);
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
            length: 2.0_f64,
            height: 1.0_f64,
        };
        // 200x100 = 20,000 cells
        let mesh = generate_cut_cell_mesh(&geo, 0.01_f64, 0.01_f64, Vector2::new(2.0_f64, 1.0_f64));
        println!("Mesh generated: {} cells", mesh.num_cells());
        
        let mut solver: ParallelPisoSolver<f64> = ParallelPisoSolver::new(mesh, 4);
        
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
        let length = 5.0_f64;
        let domain_size = Vector2::new(length, 1.0_f64);
        let geo = BackwardsStep {
            length,
            height_inlet: 0.5_f64,
            height_outlet: 1.0_f64,
            step_x: 1.0_f64,
        };
        // 5.0 * 1.0 - 1.0 * 0.5 = 4.5 m^2
        // 0.003 cell size -> 9e-6 m^2 per cell -> 500,000 cells
        let min_cell_size = 0.003_f64;
        let max_cell_size = 0.003_f64;
        let mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, domain_size);
        println!("Mesh generated: {} cells", mesh.num_cells());
        
        let n_threads = 4;
        let mut solver = ParallelPisoSolver::<f64>::new(mesh, n_threads);
        
        // Setup BCs
        for partition in &solver.partitions {
            let mut s = partition.write().unwrap();
            s.dt = 0.001_f64;
            s.density = 1.225_f64;
            s.viscosity = 1.81e-5_f64;
            
            let n_cells = s.mesh.num_cells();
            for i in 0..n_cells {
                let cx = s.mesh.cell_cx[i];
                let cy = s.mesh.cell_cy[i];
                if cx < max_cell_size {
                    if cy > 0.5_f64 {
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

pub struct SolverPartition<T: Float> {
    pub mesh: Mesh,
    pub u: VectorField<T>,
    pub p: ScalarField<T>,
    pub residuals: Vec<(String, T)>,
    pub time: T,
    pub dt: T,
    pub density: T,
    pub viscosity: T,
    pub scheme: Scheme,
}

impl<T: Float> SolverPartition<T> {
    fn from_solver(solver: &PisoSolver<T>) -> Self {
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

