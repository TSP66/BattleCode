// Batched environment for PPO: many games in parallel, one dragon turn per
// step, observations and reward components written straight into the caller's
// numpy buffers.
//
// The rules come from bc_core.hpp, which is verified against the real engine.
// Nothing here may change an outcome: this file only decides what a learner
// gets to see and when transitions close.
#pragma once

#include "bc_text.hpp"
#include "bc_bots.hpp"
#include "bc_memory.hpp"

#include <atomic>
#include <cmath>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

namespace bc {

// ---------------------------------------------------------------- layout
enum LocalChannel {
    LC_PEARL = 0, LC_PEARL_TIME, LC_NEVER_SPAWN,
    LC_SELF_HEAD, LC_SELF_BODY, LC_ALLY_HEAD, LC_ALLY_BODY, LC_ENEMY_HEAD, LC_ENEMY_BODY,
    LC_FACE_N, LC_FACE_E, LC_FACE_S, LC_FACE_W,
    LC_KELP_N, LC_KELP_E, LC_KELP_S, LC_KELP_W,
    LC_PORTAL_N, LC_PORTAL_E, LC_PORTAL_S, LC_PORTAL_W,
    LC_SELF_INDEX, LC_SELF_TAIL,
    LC_COUNT
};

enum ScalarField {
    SC_ROUND = 0, SC_LENGTH, SC_LENGTH_RAW, SC_UNITS, SC_FACE_N, SC_FACE_E, SC_FACE_S, SC_FACE_W,
    SC_HEAD_X, SC_HEAD_Y, SC_MAP_W, SC_MAP_H, SC_NUM_MSGS, SC_TEAM_B, SC_COUNT
};

// The network's scalar input: the 14 above, then mem (676) and memfar (18)
// from bc_memory.hpp, in the order imitate2.py concatenated them.
//
// The base 14 keep indices 0-13 and are never reordered or altered. That is
// what lets a checkpoint trained before memory be widened with zero columns and
// go on playing identically (train/migrate_scalars.py), so gen4 and the rest of
// the league survive the switch as frozen opponents.
constexpr int SC_TOTAL = SC_COUNT + mem_cfg::N_EXTRA;   // 708

// Reward components. The caller supplies weights; nothing is baked in.
//
// The four team components are differences of a team potential taken between
// one agent's consecutive turns, so they telescope per agent and price the
// whole team's progress rather than the dragon's own body. RW_TEAM_MAX is the
// game's actual win condition (the longest dragon), which is why splitting the
// leader has to show up as a loss somewhere.
enum RewardComp {
    RW_LENGTH_DELTA = 0, RW_PEARLS, RW_SPRINT_COST, RW_SPLIT_COST, RW_DIED,
    RW_KILLS, RW_ENEMY_DEATHS, RW_ALLY_DEATHS, RW_WIN, RW_LOSE, RW_DRAW,
    RW_FINAL_LENGTH,
    RW_SPLITS,                                   // one per split performed
    RW_TEAM_LEN, RW_TEAM_MAX, RW_FOE_LEN, RW_FOE_MAX,
    RW_PORTAL,                                   // 1 when this turn went through a portal
    // reward v6: 1 to each dragon that dies on the turn its team is wiped out,
    // when the game then goes to the enemy (not a mutual wipe-out, a draw).
    // The result used to be paid only to survivors, so an elimination cost
    // nothing on the result.
    RW_ELIMINATED,
    // reward v6: team potentials over log(1 + living units), concave so a few
    // units are worth having and many are not
    RW_TEAM_UNITS, RW_FOE_UNITS,
    RW_COUNT
};

constexpr int MAX_MSGS = 4;
// privileged critic features: our total / longest / units, theirs, round,
// and the longest-dragon margin (see VecEnv::observe)
constexpr int PRIV_COUNT = 8;
// board planes: own body, own heads, enemy body, enemy heads, pearls,
// inside-the-map, kelp on the north edge, kelp on the west edge, then the
// acting dragon's own body and head, and how soon a pearl is due per tile
// (see VecEnv::observe). Only the critic reads these, and it never ships, so
// they may say things a deployed bot cannot know.
constexpr int BOARD_CH = 11;
constexpr int BOARD_MAX = 64;
// Steps a single MOVE action may carry. The engine has no cap of its own (a
// sprint is limited by length), so replaying real games needs more:
// libbcvec_replay.so is built with -DBC_MAX_STEPS=64.
#ifndef BC_MAX_STEPS
#define BC_MAX_STEPS 8
#endif
constexpr int MAX_STEPS = BC_MAX_STEPS;
constexpr int CODEC_MOVES = 3 + 9 + 27;   // relative paths of length 1, 2, 3
constexpr int CODEC_SPLITS = 9;
constexpr int CODEC_ACTIONS = CODEC_MOVES + CODEC_SPLITS;
const int CODEC_SPLIT_K[CODEC_SPLITS] = {2, 3, 4, 5, 6, 8, 12, 16, -1};  // -1 = half

// Structured action, which is what the engine actually takes.
struct Action {
    int8_t kind;          // 0 move, 1 split, 2 suicide
    int8_t n_steps;
    int8_t dirs[MAX_STEPS];  // 0 N, 1 E, 2 S, 3 W
    int16_t split_k;
    int8_t send_sonar;
    uint32_t sonar;
};

inline char dir_char(int d) { return d == 0 ? 'N' : d == 1 ? 'E' : d == 2 ? 'S' : 'W'; }
inline int dir_index(char c) { return c == 'N' ? 0 : c == 'E' ? 1 : c == 'S' ? 2 : 3; }

// ---------------------------------------------------------------- config
struct VecConfig {
    int num_envs = 64;
    int num_threads = 8;
    uint64_t seed = 0;
    bool egocentric = true;    // rotate the window so the dragon faces north
    bool random_pearl_seed = true;
    int max_rounds = 500;
    // Discount inside the team potentials: each delta is gamma * phi(now) -
    // phi(then). 1 reproduces plain differences (reward v1/v2); set it to the
    // PPO gamma to make it potential-based shaping.
    float potential_gamma = 1.0f;
};

// One open transition waiting to be closed with its reward.
struct AgentAcc {
    uint64_t uid = 0;
    float comps[RW_COUNT] = {0};
    bool open = false;      // the agent has acted, so a transition is waiting
    bool alive = true;
    int last_len = 0;
    // the team picture as this agent last saw it, for the team potentials
    int team_len = 0, team_max = 0, foe_len = 0, foe_max = 0;
    int team_units = 0, foe_units = 0;
    bool primed = false;    // false until the agent has taken one turn
    uint64_t banked_at = ~0ull;   // env turn the potentials were last banked on
};

struct Closure {
    int32_t env;
    int64_t uid;
    float comps[RW_COUNT];
    int8_t done;   // 0 still going, 1 episode over for this agent
};

struct EpisodeStat {
    int32_t env;
    int32_t rounds;
    int32_t winner;
    int32_t a_longest, b_longest, a_units, b_units;
    int32_t map_index;
    int32_t a_deaths, b_deaths;   // dragons each team lost, whatever the cause
    int32_t a_kills, b_kills;     // enemy dragons each team's bodies or heads killed
    int32_t a_len_lost, b_len_lost;       // segments each team lost to deaths
    int32_t a_len_killed, b_len_killed;   // enemy segments each team's kills removed
    int32_t a_headon, b_headon;           // of each team's kills, those its killer died making
};
constexpr int EP_COLS = 18;

// ---------------------------------------------------------------- env
struct Env {
    Game game;
    const MapData* map = nullptr;
    int map_index = 0;
    int cursor = 0;
    bool round_open = false;
    uint64_t episode = 0;
    std::vector<AgentAcc> agents;
    std::mt19937_64 rng;
    int acting = -1;             // dragon index whose turn it is
    size_t events_seen = 0;
    // evaluation: one team played by a scripted bot, and a pinned map
    int scripted_team = -1;
    int bot_kind = 0;
    int fixed_map = -1;
    int deaths[2] = {0, 0}, kills[2] = {0, 0};
    // length each team lost to deaths, and enemy length each team's kills took
    int len_lost[2] = {0, 0}, len_killed[2] = {0, 0};
    int headon[2] = {0, 0};      // kills where the killer died in the same collision
    uint64_t turn = 0;           // dragon turns taken in this env, for banking

    // Per-dragon remembered map (bc_memory.hpp). Slots are pooled: dragon ids
    // are handed out by dragons.size() and never reused inside an episode, and
    // the vector only grows, so a 500-round game of splits and deaths would
    // leak without reclaiming. A new dragon always gets a cleared slot, which
    // is what makes a split child start empty -- the semantics the trainer and
    // the deployed bot both assume.
    std::vector<DragonMemory> mem_pool;
    std::vector<int> mem_slot;        // dragon id -> slot, -1 = none
    std::vector<int> mem_free;        // slots ready for a new dragon
    std::vector<int> mem_live;        // dragon ids currently holding a slot

    void mem_clear() {
        for (int id : mem_live)
            if (id >= 0 && id < (int)mem_slot.size()) mem_slot[(size_t)id] = -1;
        mem_live.clear();
        mem_free.clear();
        for (int i = 0; i < (int)mem_pool.size(); i++) mem_free.push_back(i);
    }

    // The acting dragon's memory, allocated and cleared on first use.
    DragonMemory& mem_for(int dragon_id, int w, int h) {
        if ((int)mem_slot.size() <= dragon_id) mem_slot.resize((size_t)dragon_id + 1, -1);
        int& slot = mem_slot[(size_t)dragon_id];
        if (slot >= 0) return mem_pool[(size_t)slot];
        if (mem_free.empty()) {                 // reclaim the dead before growing
            size_t keep = 0;
            for (size_t i = 0; i < mem_live.size(); i++) {
                int const id = mem_live[i];
                if (game.dragons[(size_t)id].alive) { mem_live[keep++] = id; continue; }
                mem_free.push_back(mem_slot[(size_t)id]);
                mem_slot[(size_t)id] = -1;
            }
            mem_live.resize(keep);
        }
        if (mem_free.empty()) {
            mem_pool.emplace_back();
            mem_free.push_back((int)mem_pool.size() - 1);
        }
        slot = mem_free.back();
        mem_free.pop_back();
        mem_live.push_back(dragon_id);
        DragonMemory& dm = mem_pool[(size_t)slot];
        dm.reset(w, h);
        return dm;
    }

    uint64_t uid_of(int dragon_id) const {
        return (episode << 12) | (uint64_t)dragon_id;
    }
};

class VecEnv {
public:
    VecEnv(const VecConfig& cfg, std::vector<MapData> maps)
        : cfg_(cfg), maps_(std::move(maps)), envs_(cfg.num_envs) {
        for (int i = 0; i < cfg_.num_envs; i++) {
            envs_[i].rng.seed(cfg_.seed * 1000003ull + (uint64_t)i * 7919ull + 17ull);
            envs_[i].episode = (uint64_t)i << 32;
        }
        closures_.reserve((size_t)cfg_.num_envs * 8);
        start_workers();
    }

    ~VecEnv() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    int num_envs() const { return cfg_.num_envs; }

    // Map sampling weights, one per map; takes effect at each env's next episode.
    void set_map_weights(const double* w) {
        map_cdf_.assign(maps_.size(), 0.0);
        double acc = 0.0;
        for (size_t i = 0; i < maps_.size(); i++) {
            acc += w[i] > 0.0 ? w[i] : 0.0;
            map_cdf_[i] = acc;
        }
        if (acc <= 0.0) map_cdf_.clear();
    }

    // Evaluation hooks, per env: pin a map (-1 = sample), and hand one team
    // (-1 = none) to a scripted bot. Also take effect at the next episode, so
    // call reset() after setting them.
    void set_potential_gamma(float g) { cfg_.potential_gamma = g; }

    void set_env_opponent(int env_index, int team, int bot_kind, int fixed_map) {
        Env& e = envs_[env_index];
        e.scripted_team = team;
        e.bot_kind = bot_kind;
        e.fixed_map = fixed_map;
    }

    // ---- buffers the caller owns
    void bind(float* local, float* scalar, uint32_t* msgs, uint8_t* mask,
              int64_t* uid, int32_t* dragon_id, int8_t* team, int32_t* round_out) {
        b_local_ = local; b_scalar_ = scalar; b_msgs_ = msgs; b_mask_ = mask;
        b_uid_ = uid; b_dragon_ = dragon_id; b_team_ = team; b_round_ = round_out;
    }

    // Optional: privileged global features per row (PRIV_COUNT floats), for a
    // critic that never ships. Nothing is written unless this is bound.
    void bind_priv(float* priv) { b_priv_ = priv; }
    // Optional: the whole board, BOARD_CH planes of BOARD_MAX x BOARD_MAX
    // (uint8, map in the top-left corner, the rest zero), relative to the
    // acting dragon's team. For offline critic studies; costs a full write
    // per step, so leave it unbound in training.
    void bind_board(uint8_t* board) { b_board_ = board; }
    // Optional: the remembered map as wide_cfg::N_WIDE floats per row, two
    // stacked scales of six planes in the acting dragon's own frame
    // (bc_memory.hpp `wide`). The 708-scalar nets never ask for it.
    void bind_wide(float* wide) { b_wide_ = wide; }

    void reset() {
        closures_.clear();
        episodes_.clear();
        run_parallel([&](int i) {
            begin_episode(envs_[i], i);
            advance(envs_[i], i);
            observe(envs_[i], i);
        });
    }

    // Applies one action per env, advances to the next acting dragon and
    // writes the new observations.
    void step(const Action* actions) {
        closures_.clear();
        episodes_.clear();
        std::vector<std::vector<Closure>> per_thread(cfg_.num_threads);
        std::vector<std::vector<EpisodeStat>> stats(cfg_.num_threads);
        run_parallel_t([&](int i, int t) {
            Env& e = envs_[i];
            apply(e, i, actions[i], per_thread[t]);
            advance(e, i, &per_thread[t], &stats[t]);
            observe(e, i);
        });
        for (auto& v : per_thread) closures_.insert(closures_.end(), v.begin(), v.end());
        for (auto& v : stats) episodes_.insert(episodes_.end(), v.begin(), v.end());
    }

    const std::vector<Closure>& closures() const { return closures_; }

    // Test hooks: the acting dragon, and the protocol block it would be sent.
    int acting_dragon_id(int env_index) const {
        const Env& e = envs_[env_index];
        return e.acting < 0 ? -1 : e.game.dragons[e.acting].id;
    }
    std::string round_block(int env_index) const {
        const Env& e = envs_[env_index];
        return e.acting < 0 ? std::string() : render_round_block(e.game, e.acting);
    }
    bool finished(int env_index) const { return envs_[env_index].game.finished; }
    const std::vector<EpisodeStat>& episodes() const { return episodes_; }

    // Turns a codec action id into the structured action, using the acting
    // dragon's own length for the split sizes.
    Action decode(int env_index, int action_id) const {
        const Env& e = envs_[env_index];
        const Dragon& d = e.game.dragons[e.acting];
        return decode_for(d, action_id);
    }

    // What each codec move would do right now, tried on a copy of the game
    // (the env itself is untouched). Used to label replays, never to act.
    // out[0..2] = enemy dragons alive, the longest enemy's length, our team's
    // dragons alive, all before the move; then PROBE_FIELDS per move id
    // 0..CODEC_MOVES-1 (see PROBE_*), all zero for a move the mask forbids.
    //
    // "Trapped" is conservative: every way forward is kelp or a body segment
    // that cannot move out of the way first (not a head, not someone else's
    // tail; its own tail still kills it), and it is too short to split, or
    // its team is full.
    enum { PROBE_OUR_ALIVE = 0, PROBE_KILLED, PROBE_LONGEST_KILLED, PROBE_ENEMY_ALIVE,
           PROBE_TRAPPED, PROBE_LONGEST_TRAPPED, PROBE_KILLED_LEN, PROBE_AREA, PROBE_FIELDS };
    static constexpr int PROBE_AREA_CAP = 128;

    // Free cells reachable from the acting dragon's head (not counting it),
    // through edges a step can cross, up to PROBE_AREA_CAP. Every body is a
    // wall, tails included: a lower bound on the room the dragon has.
    static int free_area(const Game& g, int di) {
        const MapData& m = *g.map;
        const Dragon& d = g.dragons[di];
        std::vector<int16_t> queue;
        std::vector<char> seen(m.area(), 0);
        queue.push_back(d.head());
        seen[d.head()] = 1;
        static const char DIRS[4] = {'N', 'E', 'S', 'W'};
        int n = 0;
        for (size_t q = 0; q < queue.size() && n < PROBE_AREA_CAP; q++) {
            const int c = queue[q];
            for (char dir : DIRS) {
                int nx, ny;
                if (!tile_after_step(m, c % m.w, c / m.w, dir, nx, ny)) continue;
                const int t = m.idx(nx, ny);
                if (seen[t] || g.owner[t] >= 0) continue;
                seen[t] = 1;
                queue.push_back((int16_t)t);
                if (++n >= PROBE_AREA_CAP) break;
            }
        }
        return n;
    }

    static bool trapped(const Game& g, int j) {
        const Dragon& d = g.dragons[j];
        const MapData& m = *g.map;
        if (!d.alive) return false;
        if (d.len >= 4 && g.alive[d.team] < m.unit_limit) return false;   // can split
        const int x = d.head() % m.w, y = d.head() / m.w;
        const char back = Game::opposite(d.facing);
        static const char DIRS[4] = {'N', 'E', 'S', 'W'};
        for (char dir : DIRS) {
            if (dir == back) continue;
            int nx, ny;
            if (!tile_after_step(m, x, y, dir, nx, ny)) continue;          // kelp
            const int t = m.idx(nx, ny);
            const int16_t occ = g.owner[t];
            if (occ < 0) return false;
            if (occ == (int16_t)j) continue;                               // own body or tail
            if (g.head_at[t]) return false;                                // may move away
            if (g.dragons[occ].tail() == (int16_t)t) return false;         // may vacate
        }
        return true;
    }

    void probe(int env_index, int32_t* out) const {
        const Env& e = envs_[env_index];
        const Game& g0 = e.game;
        const int di = e.acting;
        const int me = g0.dragons[di].team, foe = 1 - me;
        int longest = 0;
        for (const Dragon& d : g0.dragons)
            if (d.alive && d.team == foe) longest = std::max(longest, d.len);
        std::vector<char> was_trapped(g0.dragons.size(), 0);
        for (size_t j = 0; j < g0.dragons.size(); j++)
            if (g0.dragons[j].alive && g0.dragons[j].team == foe) was_trapped[j] = trapped(g0, (int)j);
        out[0] = g0.alive[foe];
        out[1] = longest;
        out[2] = g0.alive[me];
        const uint8_t* mask = b_mask_ + (size_t)env_index * CODEC_ACTIONS;
        for (int id = 0; id < CODEC_MOVES; id++) {
            int32_t* o = out + 3 + id * PROBE_FIELDS;
            for (int k = 0; k < PROBE_FIELDS; k++) o[k] = 0;
            if (!mask[id]) continue;
            Game g = g0;
            g.record_events = false;
            const Action a = decode_for(g.dragons[di], id);
            char dirs[MAX_STEPS];
            for (int s = 0; s < a.n_steps; s++) dirs[s] = dir_char(a.dirs[s]);
            g.move(di, dirs, a.n_steps);
            o[PROBE_OUR_ALIVE] = g.dragons[di].alive;
            o[PROBE_AREA] = g.dragons[di].alive ? free_area(g, di) : 0;
            o[PROBE_ENEMY_ALIVE] = g.alive[foe];
            for (size_t j = 0; j < g.dragons.size(); j++) {
                const Dragon& before = g0.dragons[j];
                if (!before.alive || before.team != foe) continue;
                const bool is_longest = before.len == longest;
                if (!g.dragons[j].alive) {
                    o[PROBE_KILLED]++;
                    o[PROBE_KILLED_LEN] += before.len;
                    if (is_longest) o[PROBE_LONGEST_KILLED] = 1;
                } else if (!was_trapped[j] && trapped(g, (int)j)) {
                    o[PROBE_TRAPPED]++;
                    if (is_longest) o[PROBE_LONGEST_TRAPPED] = 1;
                }
            }
        }
    }

    // Deaths the last action caused in env_index: (dragon id, reason, killer
    // id or -1, team) rows, at most cap. Read right after a step.
    int last_deaths(int env_index, int32_t* out, int cap) const {
        const Env& e = envs_[env_index];
        int n = 0;
        for (const Event& ev : e.game.events) {
            if (ev.kind != EV_DEATH) continue;
            if (n < cap) {
                out[4 * n] = ev.a;
                out[4 * n + 1] = ev.b;
                out[4 * n + 2] = ev.d;
                out[4 * n + 3] = e.game.dragons[ev.a].team;
            }
            n++;
        }
        return n;
    }

    static Action decode_for(const Dragon& d, int action_id) {
        Action a{};
        a.kind = 2;
        a.n_steps = 0;
        a.split_k = 0;
        a.send_sonar = 0;
        a.sonar = 0;
        if (action_id < 0 || action_id >= CODEC_ACTIONS) return a;
        if (action_id < CODEC_MOVES) {
            int n, rest;
            if (action_id < 3) { n = 1; rest = action_id; }
            else if (action_id < 12) { n = 2; rest = action_id - 3; }
            else { n = 3; rest = action_id - 12; }
            a.kind = 0;
            a.n_steps = (int8_t)n;
            // digits are read low first: step 0 is the least significant turn
            int facing = dir_index(d.facing);
            for (int i = 0; i < n; i++) {
                const int turn = rest % 3;
                rest /= 3;
                facing = turn == 0 ? facing : (turn == 1 ? (facing + 3) % 4 : (facing + 1) % 4);
                a.dirs[i] = (int8_t)facing;
            }
            return a;
        }
        const int k = CODEC_SPLIT_K[action_id - CODEC_MOVES];
        a.kind = 1;
        a.split_k = (int16_t)(k < 0 ? d.len / 2 : k);
        return a;
    }

private:
    // ---- episode lifecycle
    void begin_episode(Env& e, int index) {
        if (e.fixed_map >= 0 && e.fixed_map < (int)maps_.size()) {
            e.map_index = e.fixed_map;
        } else if (!map_cdf_.empty()) {
            const double u = std::uniform_real_distribution<double>(0.0, map_cdf_.back())(e.rng);
            e.map_index = (int)(std::upper_bound(map_cdf_.begin(), map_cdf_.end(), u) -
                                map_cdf_.begin());
            if (e.map_index >= (int)maps_.size()) e.map_index = (int)maps_.size() - 1;
        } else {
            e.map_index = (int)(e.rng() % maps_.size());
        }
        e.map = &maps_[e.map_index];
        const uint32_t seed = cfg_.random_pearl_seed ? (uint32_t)e.rng()
                                                     : MT19937::DEFAULT_SEED;
        e.game.reset(*e.map, seed);
        e.game.record_events = true;
        e.cursor = 0;
        e.round_open = false;
        e.events_seen = 0;
        e.deaths[0] = e.deaths[1] = e.kills[0] = e.kills[1] = 0;
        e.len_lost[0] = e.len_lost[1] = e.len_killed[0] = e.len_killed[1] = 0;
        e.headon[0] = e.headon[1] = 0;
        e.episode++;
        e.mem_clear();          // every dragon of the new episode starts blank
        e.agents.assign(e.game.dragons.size(), AgentAcc());
        for (size_t i = 0; i < e.agents.size(); i++) {
            e.agents[i].uid = e.uid_of(e.game.dragons[i].id);
            e.agents[i].last_len = e.game.dragons[i].len;
        }
        (void)index;
    }

    void ensure_agents(Env& e) {
        while (e.agents.size() < e.game.dragons.size()) {
            AgentAcc acc;
            const size_t i = e.agents.size();
            acc.uid = e.uid_of(e.game.dragons[i].id);
            acc.last_len = e.game.dragons[i].len;
            e.agents.push_back(acc);
        }
    }

    // ---- applying one action
    void apply(Env& e, int index, const Action& a, std::vector<Closure>& out) {
        if (e.acting < 0) return;
        const int di = e.acting;
        Dragon& d = e.game.dragons[di];
        d.inbox.clear();
        for (int k = 0; k < SONAR_ECHO_KINDS; k++) d.echo[k] = 0;
        e.game.events.clear();
        e.events_seen = 0;

        const int before_len = d.len;
        const int64_t portals_before = e.game.stats[ST_PORTAL_STEP];
        if (a.kind == 0 && a.n_steps > 0) {
            char dirs[MAX_STEPS];
            const int n = a.n_steps > MAX_STEPS ? MAX_STEPS : a.n_steps;
            for (int i = 0; i < n; i++) dirs[i] = dir_char(a.dirs[i]);
            e.game.move(di, dirs, n);
        } else if (a.kind == 1) {
            e.game.split(di, a.split_k);
        } else {
            e.game.kill(di, DEATH_ACTION);
        }
        if (a.send_sonar && e.game.dragons[di].alive) e.game.cast_sonar(di, a.sonar);

        ensure_agents(e);
        e.turn++;
        if (e.game.stats[ST_PORTAL_STEP] != portals_before) e.agents[di].comps[RW_PORTAL] += 1.0f;
        harvest(e, index, di, before_len, out);
        e.cursor = di + 1;
    }

    static void team_stats(const Env& e, uint8_t team, int& total, int& longest, int& units) {
        total = 0;
        longest = 0;
        units = 0;
        for (const Dragon& d : e.game.dragons) {
            if (!d.alive || d.team != team) continue;
            total += d.len;
            if (d.len > longest) longest = d.len;
            units++;
        }
    }

    static float units_phi(int units) { return std::log1p((float)units); }

    // Pays an agent the change in the team potentials since its own last turn.
    // Because it is a difference of a state function it telescopes, so the
    // weights shape behaviour without changing which policy is optimal. It is
    // also how a dragon that dies usefully gets paid: its final transition
    // still sees the enemy length it took with it.
    void add_team_delta(Env& e, int agent) {
        AgentAcc& acc = e.agents[agent];
        // at most once per turn: with a discount, banking the same state twice
        // would add a spurious (gamma - 1) * phi
        if (acc.banked_at == e.turn) return;
        acc.banked_at = e.turn;
        const uint8_t team = e.game.dragons[agent].team;
        int tl, tm, tu, fl, fm, fu;
        team_stats(e, team, tl, tm, tu);
        team_stats(e, (uint8_t)(1 - team), fl, fm, fu);
        if (acc.primed) {
            // gamma * phi(now) - phi(then). phi is never zeroed at a death: a
            // dead dragon keeps the team position it left behind, which is how
            // a sacrifice gets paid for the trade it made (reward v3).
            const float g = cfg_.potential_gamma;
            acc.comps[RW_TEAM_LEN] += g * (float)tl - (float)acc.team_len;
            acc.comps[RW_TEAM_MAX] += g * (float)tm - (float)acc.team_max;
            acc.comps[RW_FOE_LEN] += g * (float)fl - (float)acc.foe_len;
            acc.comps[RW_FOE_MAX] += g * (float)fm - (float)acc.foe_max;
            acc.comps[RW_TEAM_UNITS] += g * units_phi(tu) - units_phi(acc.team_units);
            acc.comps[RW_FOE_UNITS] += g * units_phi(fu) - units_phi(acc.foe_units);
        }
        acc.team_len = tl;
        acc.team_max = tm;
        acc.foe_len = fl;
        acc.foe_max = fm;
        acc.team_units = tu;
        acc.foe_units = fu;
        acc.primed = true;
    }

    // Turns this turn's events into reward components for everyone affected.
    void harvest(Env& e, int index, int actor, int before_len, std::vector<Closure>& out) {
        AgentAcc& me = e.agents[actor];
        // Both of these have to be banked before the event loop: an actor that
        // killed itself this turn is closed inside that loop, and anything
        // added afterwards would land in a zeroed accumulator and be lost.
        const int after_len = e.game.dragons[actor].alive ? e.game.dragons[actor].len : 0;
        me.comps[RW_LENGTH_DELTA] += (float)(after_len - before_len);
        if (e.game.dragons[actor].alive) me.last_len = after_len;
        add_team_delta(e, actor);

        for (const Event& ev : e.game.events) {
            switch (ev.kind) {
                case EV_PEARL_EATEN:
                    me.comps[RW_PEARLS] += 1.0f;
                    break;
                case EV_SPLIT: {
                    me.comps[RW_SPLIT_COST] += (float)ev.c;   // segments handed over
                    me.comps[RW_SPLITS] += 1.0f;              // flat, per split
                    break;
                }
                case EV_DEATH: {
                    const int victim = find_index(e, ev.a);
                    if (victim < 0) break;
                    AgentAcc& va = e.agents[victim];
                    va.comps[RW_DIED] += 1.0f;
                    va.comps[RW_FINAL_LENGTH] += (float)ev.c;
                    // Death costs a dragon everything it grew. The actor
                    // already banked that above via after_len == 0; anyone
                    // killed on someone else's turn has to be charged here, or
                    // length_delta would mean two different things.
                    if (victim != actor) va.comps[RW_LENGTH_DELTA] -= (float)ev.c;
                    va.alive = false;
                    const uint8_t vteam = e.game.dragons[victim].team;
                    e.deaths[vteam]++;
                    e.len_lost[vteam] += ev.c;
                    if (ev.d >= 0) {
                        const int killer = find_index(e, ev.d);
                        if (killer >= 0 && e.game.dragons[killer].team != vteam) {
                            // Only a killer that survives is paid. In a head-on
                            // both die, and paying both made mutual kamikaze
                            // profitable (reward v3's first launch: +0.75 kill
                            // -0.25 died each). A head-on is judged by the
                            // trade alone, through the zero-sum team terms.
                            if (e.game.dragons[killer].alive) e.agents[killer].comps[RW_KILLS] += 1.0f;
                            else e.headon[1 - vteam]++;
                            e.kills[1 - vteam]++;
                            e.len_killed[1 - vteam] += ev.c;
                        }
                    }
                    for (size_t i = 0; i < e.agents.size(); i++) {
                        if (!e.agents[i].alive) continue;
                        if (e.game.dragons[i].team == vteam) e.agents[i].comps[RW_ALLY_DEATHS] += 1.0f;
                        else e.agents[i].comps[RW_ENEMY_DEATHS] += 1.0f;
                    }
                    break;
                }
                default:
                    break;
            }
        }
        // Close the dead only once every death this turn is credited. In a
        // head-on both die; closing the first victim inside the loop above
        // lost the kill it is credited with by the second death.
        for (const Event& ev : e.game.events) {
            if (ev.kind != EV_DEATH) continue;
            const int victim = find_index(e, ev.a);
            if (victim < 0 || !e.agents[victim].open) continue;
            add_team_delta(e, victim);   // credit what it died for
            // A wipe-out ends the game with this round, but the result is only
            // settled when the round does: the enemy can still be wiped out
            // later in it, which is a draw. Hold the transition open so
            // finish_episode can pay what it turns out to be.
            if (e.game.alive[e.game.dragons[victim].team] == 0) continue;
            close(e, index, victim, 1, out);
        }
    }

    static int find_index(Env& e, int dragon_id) {
        // ids are assigned in creation order, so the index is the id
        if (dragon_id < 0 || dragon_id >= (int)e.game.dragons.size()) return -1;
        return dragon_id;
    }

    void close(Env& e, int index, int agent, int8_t done, std::vector<Closure>& out) {
        AgentAcc& acc = e.agents[agent];
        if (e.game.dragons[agent].team == e.scripted_team) {   // nobody learns from a bot
            memset(acc.comps, 0, sizeof acc.comps);
            acc.open = false;
            return;
        }
        Closure c;
        c.env = index;
        c.uid = (int64_t)acc.uid;
        memcpy(c.comps, acc.comps, sizeof c.comps);
        c.done = done;
        out.push_back(c);
        memset(acc.comps, 0, sizeof acc.comps);
        acc.open = false;
    }

    // ---- advancing to the next dragon that has to act
    void advance(Env& e, int index, std::vector<Closure>* out = nullptr,
                 std::vector<EpisodeStat>* stats = nullptr) {
        for (;;) {
            while (!e.game.finished) {
                if (!e.round_open) {
                    e.game.pearl_tick();
                    e.round_open = true;
                    e.cursor = 0;
                }
                ensure_agents(e);
                while (e.cursor < (int)e.game.dragons.size()) {
                    if (e.game.dragons[e.cursor].alive &&
                        (int)e.game.dragons[e.cursor].team == e.scripted_team) {
                        // a bot's turn is played here and never seen outside
                        e.acting = e.cursor;
                        const Action a = bot_action(e, e.acting);
                        if (out) apply(e, index, a, *out);
                        else { scratch_.clear(); apply(e, index, a, scratch_); }
                        continue;                 // apply moved the cursor on
                    }
                    if (e.game.dragons[e.cursor].alive) {
                        e.acting = e.cursor;
                        if (out) {
                            AgentAcc& acc = e.agents[e.acting];
                            if (acc.open) close(e, index, e.acting, 0, *out);
                            acc.open = true;
                        } else {
                            e.agents[e.acting].open = true;
                        }
                        return;
                    }
                    e.cursor++;
                }
                e.game.settle(e.game.round + 1 >= cfg_.max_rounds);
                if (e.game.finished) break;
                e.game.round++;
                e.round_open = false;
            }
            // episode over: pay out the result and start again
            finish_episode(e, index, out, stats);
            begin_episode(e, index);
        }
    }

    void finish_episode(Env& e, int index, std::vector<Closure>* out,
                        std::vector<EpisodeStat>* stats) {
        const int winner = e.game.winner;
        for (size_t i = 0; i < e.agents.size(); i++) {
            AgentAcc& acc = e.agents[i];
#ifdef BC_DEBUG_REWARD
            if (!acc.open && (acc.comps[RW_DIED] != 0 || acc.comps[RW_KILLS] != 0))
                std::fprintf(stderr, "UNPAID dragon %zu primed %d died %g kills %g\n", i,
                             (int)acc.primed, acc.comps[RW_DIED], acc.comps[RW_KILLS]);
#endif
            if (!acc.open) continue;
            const Dragon& d = e.game.dragons[i];
            // the dead banked their potentials when they died
            if (d.alive) add_team_delta(e, (int)i);
            if (d.alive) {
                acc.comps[RW_FINAL_LENGTH] += (float)d.len;
                if (winner < 0) acc.comps[RW_DRAW] += 1.0f;
                else if (winner == (int)d.team) acc.comps[RW_WIN] += 1.0f;
                else acc.comps[RW_LOSE] += 1.0f;
            } else {
                // held open by harvest: it died in the round its team was
                // wiped out, so the game is a draw or an elimination
                if (winner < 0) acc.comps[RW_DRAW] += 1.0f;
                else acc.comps[RW_ELIMINATED] += 1.0f;
            }
            if (out) close(e, index, (int)i, 1, *out);
            else acc.open = false;
        }
        if (stats) {
            EpisodeStat s{};
            s.env = index;
            s.rounds = e.game.round + 1;
            s.winner = winner;
            s.map_index = e.map_index;
            s.a_deaths = e.deaths[0];
            s.b_deaths = e.deaths[1];
            s.a_kills = e.kills[0];
            s.b_kills = e.kills[1];
            s.a_len_lost = e.len_lost[0];
            s.b_len_lost = e.len_lost[1];
            s.a_len_killed = e.len_killed[0];
            s.b_len_killed = e.len_killed[1];
            s.a_headon = e.headon[0];
            s.b_headon = e.headon[1];
            for (const Dragon& d : e.game.dragons) {
                if (!d.alive) continue;
                if (d.team == 0) { s.a_units++; s.a_longest = std::max(s.a_longest, d.len); }
                else { s.b_units++; s.b_longest = std::max(s.b_longest, d.len); }
            }
            stats->push_back(s);
        }
    }

    Action bot_action(Env& e, int di) {
        // one brain per env would be tidier, but its scratch is only a few KB
        // and envs run on one thread at a time, so a thread-local one is enough
        thread_local BotBrain brain;
        const BotMove mv = brain.act(e.game, di, e.bot_kind, e.rng);
        Action a{};
        a.kind = (int8_t)mv.kind;
        a.n_steps = mv.kind == 0 ? 1 : 0;
        a.dirs[0] = (int8_t)mv.dir;
        a.split_k = (int16_t)mv.split_k;
        return a;
    }

    // ---- observations
    void observe(Env& e, int index);
    void fill_mask(Env& e, int index);

    // ---- threading
    void start_workers() {
        for (int t = 0; t < cfg_.num_threads; t++)
            workers_.emplace_back([this, t] { worker(t); });
    }

    void worker(int t) {
        for (;;) {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stop_ || generation_ > seen_[t]; });
            if (stop_) return;
            seen_[t] = generation_;
            auto job = job_;
            lock.unlock();
            const int n = cfg_.num_envs;
            for (int i = t; i < n; i += cfg_.num_threads) job(i, t);
            if (--pending_ == 0) {
                std::lock_guard<std::mutex> done_lock(mu_);
                done_cv_.notify_all();
            }
        }
    }

    void run_parallel(const std::function<void(int)>& fn) {
        run_parallel_t([&](int i, int) { fn(i); });
    }

    void run_parallel_t(const std::function<void(int, int)>& fn) {
        if (cfg_.num_threads <= 1) {
            for (int i = 0; i < cfg_.num_envs; i++) fn(i, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            job_ = fn;
            pending_ = cfg_.num_threads;
            generation_++;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [&] { return pending_ == 0; });
    }

    VecConfig cfg_;
    std::vector<MapData> maps_;
    std::vector<Env> envs_;
    std::vector<Closure> closures_;
    std::vector<EpisodeStat> episodes_;
    std::vector<double> map_cdf_;
    static thread_local std::vector<Closure> scratch_;

    float* b_priv_ = nullptr;
    uint8_t* b_board_ = nullptr;
    float* b_wide_ = nullptr;
    float* b_local_ = nullptr;
    float* b_scalar_ = nullptr;
    uint32_t* b_msgs_ = nullptr;
    uint8_t* b_mask_ = nullptr;
    int64_t* b_uid_ = nullptr;
    int32_t* b_dragon_ = nullptr;
    int8_t* b_team_ = nullptr;
    int32_t* b_round_ = nullptr;

    std::vector<std::thread> workers_;
    std::function<void(int, int)> job_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::atomic<int> pending_{0};
    uint64_t generation_ = 0;
    uint64_t seen_[256] = {0};
    bool stop_ = false;
};

inline thread_local std::vector<Closure> VecEnv::scratch_;

}  // namespace bc
