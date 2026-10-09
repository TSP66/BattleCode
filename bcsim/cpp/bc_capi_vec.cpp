// C ABI for the batched environment.
#include "bc_obs.hpp"

#include <cstring>

using namespace bc;

extern "C" {

struct VecHandle {
    VecEnv* env;
    std::vector<Action> actions;
};

// maps: one map file per entry, joined by a NUL byte, count given separately.
void* bcv_create(const char* maps_blob, const int* map_lengths, int num_maps,
                 int num_envs, int num_threads, unsigned long long seed,
                 int egocentric, int random_pearl_seed, int max_rounds,
                 char* err, int errcap) {
    std::vector<MapData> maps;
    const char* p = maps_blob;
    for (int i = 0; i < num_maps; i++) {
        MapData m;
        std::string message;
        if (!load_map(std::string(p, map_lengths[i]), m, message)) {
            snprintf(err, errcap, "map %d: %s", i, message.c_str());
            return nullptr;
        }
        if (m.dragons.empty()) {
            snprintf(err, errcap, "map %d has no dragons", i);
            return nullptr;
        }
        maps.push_back(std::move(m));
        p += map_lengths[i];
    }
    if (maps.empty()) {
        snprintf(err, errcap, "no maps given");
        return nullptr;
    }
    VecConfig cfg;
    cfg.num_envs = num_envs;
    cfg.num_threads = num_threads < 1 ? 1 : (num_threads > 128 ? 128 : num_threads);
    cfg.seed = seed;
    cfg.egocentric = egocentric != 0;
    cfg.random_pearl_seed = random_pearl_seed != 0;
    cfg.max_rounds = max_rounds;

    auto* h = new VecHandle();
    h->env = new VecEnv(cfg, std::move(maps));
    h->actions.resize(num_envs);
    err[0] = 0;
    return h;
}

void bcv_bind_priv(void* p, float* priv) { ((VecHandle*)p)->env->bind_priv(priv); }
int bcv_priv_count() { return PRIV_COUNT; }
int bcv_priv_base() { return PRIV_BASE; }
// Broadcast sonar in all four directions every turn and speak protocol 3.
// A setter rather than another bcv_create argument: the signature is shared
// with the replay and privileged builds and with every existing caller.
void bcv_set_sonar(void* p, int on) { ((VecHandle*)p)->env->set_sonar(on != 0); }

void bcv_bind_board(void* p, uint8_t* board) { ((VecHandle*)p)->env->bind_board(board); }
void bcv_board_shape(int* out) { out[0] = BOARD_CH; out[1] = BOARD_MAX; }
// the critic view (bc_vec.hpp, cview_stride): 0 bound, -1 refused (w even or out of 1..63)
int bcv_bind_cview(void* p, uint8_t* buf, int w) { return ((VecHandle*)p)->env->bind_cview(buf, w) ? 0 : -1; }
// out: bit bytes, byte-plane offset, coarse offset, row stride, CV_CH, CV_BITS, coarse side
void bcv_cview_layout(int w, int* out) {
    out[0] = (int)cview_bits_bytes(w); out[1] = (int)cview_bits_bytes(w); out[2] = (int)cview_coarse_off(w);
    out[3] = (int)cview_stride(w); out[4] = CV_CH; out[5] = CV_BITS; out[6] = CV_COARSE;
}
// the board plane each view plane is taken from (CV_CH entries)
void bcv_cview_src(int* out) { for (int p = 0; p < CV_CH; p++) out[p] = CV_SRC[p]; }

void bcv_bind_wide(void* p, float* wide) { ((VecHandle*)p)->env->bind_wide(wide); }
void bcv_wide_shape(int* out) { out[0] = 2 * wide_cfg::CH; out[1] = wide_cfg::SIDE; }
void bcv_bind_grid(void* p, float* grid) { ((VecHandle*)p)->env->bind_grid(grid); }
void bcv_grid_shape(int* out) { out[0] = grid_cfg::CH; out[1] = grid_cfg::G; }

// Sonar v2 (bc_sonar2.hpp). 1 when the library was built with BC_SONAR2, i.e. its
// grid has the report planes and its dragons read the team packet.
int bcv_sonar2_build() {
#ifdef BC_SONAR2
    return 1;
#else
    return 0;
#endif
}
// 1 when built with BC_PORTALREP: the packet's probability fields carry the portal report
// instead, and grid channels 39-42 are the portal planes (bc_memory.hpp grid_cfg PT_*)
int bcv_portal_build() {
#ifdef BC_PORTALREP
    return 1;
#else
    return 0;
#endif
}
void bcv_set_sonar2(void* p, int env_index, int mask) { ((VecHandle*)p)->env->set_sonar2(env_index, mask); }
void bcv_bind_intent(void* p, float* intent) { ((VecHandle*)p)->env->bind_intent(intent); }
void bcv_restart_env(void* p, int env_index) { ((VecHandle*)p)->env->restart(env_index); }
unsigned long long bcv_s2_preview(void* p, int env_index) { return ((VecHandle*)p)->env->s2_preview(env_index); }

void bcv_destroy(void* p) {
    auto* h = (VecHandle*)p;
    delete h->env;
    delete h;
}

void bcv_bind(void* p, float* local, float* scalar, unsigned long long* msgs,
              unsigned char* mask, long long* uid, int* dragon_id, signed char* team,
              int* round_out) {
    ((VecHandle*)p)->env->bind(local, scalar, (uint64_t*)msgs, mask, (int64_t*)uid,
                               (int32_t*)dragon_id, (int8_t*)team, (int32_t*)round_out);
}

void bcv_reset(void* p) { ((VecHandle*)p)->env->reset(); }

// Structured actions, one per env, as flat arrays.
// `send_dirs` is one bitmask per env over {N,E,S,W}; `sonar` is SONAR_DIRS
// payloads per env, at full 64-bit width. Both may be null for "cast nothing".
static void set_sonar(Action& a, const unsigned char* send_dirs,
                      const unsigned long long* sonar, const signed char* protocol,
                      int i) {
    a.send_dirs = send_dirs ? send_dirs[i] : 0u;
    a.protocol = protocol ? protocol[i] : 0;
    for (int k = 0; k < SONAR_DIRS; k++)
        a.sonar_dir[k] = sonar ? sonar[(size_t)i * SONAR_DIRS + k] : 0ull;
}

void bcv_step(void* p, const signed char* kind, const signed char* n_steps,
              const signed char* dirs, const short* split_k,
              const unsigned char* send_dirs, const unsigned long long* sonar,
              const signed char* protocol) {
    auto* h = (VecHandle*)p;
    const int n = h->env->num_envs();
    for (int i = 0; i < n; i++) {
        Action& a = h->actions[i];
        a.kind = kind[i];
        a.n_steps = n_steps[i];
        memcpy(a.dirs, dirs + (size_t)i * MAX_STEPS, MAX_STEPS);
        a.split_k = split_k[i];
        set_sonar(a, send_dirs, sonar, protocol, i);
    }
    h->env->step(h->actions.data());
}

// bcv_step, plus the codec id each structured action stands for (-1 unknown): the action history
// (BC_AHIST; VecEnv::note_action) of a replayed game, whose moves come in as structured actions.
void bcv_step_ids(void* p, const signed char* kind, const signed char* n_steps,
                  const signed char* dirs, const short* split_k,
                  const unsigned char* send_dirs, const unsigned long long* sonar,
                  const signed char* protocol, const int* ids) {
    auto* h = (VecHandle*)p;
    const int n = h->env->num_envs();
    for (int i = 0; i < n; i++) {
        Action& a = h->actions[i];
        a.kind = kind[i];
        a.n_steps = n_steps[i];
        memcpy(a.dirs, dirs + (size_t)i * MAX_STEPS, MAX_STEPS);
        a.split_k = split_k[i];
        set_sonar(a, send_dirs, sonar, protocol, i);
    }
    h->env->step(h->actions.data(), ids);
}

// Codec actions: one id per env, plus the optional sonar payloads.
void bcv_step_codec(void* p, const int* action_ids, const unsigned char* send_dirs,
                    const unsigned long long* sonar, const signed char* protocol) {
    auto* h = (VecHandle*)p;
    const int n = h->env->num_envs();
    for (int i = 0; i < n; i++) {
        Action a = h->env->decode(i, action_ids[i]);
        set_sonar(a, send_dirs, sonar, protocol, i);
        h->actions[i] = a;
    }
    h->env->step(h->actions.data(), action_ids);
}

// The true, uncapped count of payloads handed to each acting dragon.
void bcv_bind_num_msgs(void* p, int* n) { ((VecHandle*)p)->env->bind_num_msgs(n); }

// Closed transitions from the last step: env index, uid, reward components, done.
int bcv_closures(void* p, int* env_out, long long* uid_out, float* comps_out,
                 signed char* done_out, int cap) {
    auto* h = (VecHandle*)p;
    const auto& cl = h->env->closures();
    const int n = (int)cl.size() < cap ? (int)cl.size() : cap;
    for (int i = 0; i < n; i++) {
        env_out[i] = cl[i].env;
        uid_out[i] = cl[i].uid;
        memcpy(comps_out + (size_t)i * RW_COUNT, cl[i].comps, sizeof(float) * RW_COUNT);
        done_out[i] = cl[i].done;
    }
    return (int)cl.size();
}

int bcv_episodes(void* p, int* out, int cap) {
    auto* h = (VecHandle*)p;
    const auto& ep = h->env->episodes();
    const int n = (int)ep.size() < cap ? (int)ep.size() : cap;
    for (int i = 0; i < n; i++) {
        int* row = out + (size_t)i * EP_COLS;
        row[0] = ep[i].env;
        row[1] = ep[i].rounds;
        row[2] = ep[i].winner;
        row[3] = ep[i].a_longest;
        row[4] = ep[i].b_longest;
        row[5] = ep[i].a_units;
        row[6] = ep[i].b_units;
        row[7] = ep[i].map_index;
        row[8] = ep[i].a_deaths;
        row[9] = ep[i].b_deaths;
        row[10] = ep[i].a_kills;
        row[11] = ep[i].b_kills;
        row[12] = ep[i].a_len_lost;
        row[13] = ep[i].b_len_lost;
        row[14] = ep[i].a_len_killed;
        row[15] = ep[i].b_len_killed;
        row[16] = ep[i].a_headon;
        row[17] = ep[i].b_headon;
    }
    return (int)ep.size();
}

// Outcomes of every codec move for the acting dragon (VecEnv::probe).
void bcv_probe(void* p, int env_index, int* out) {
    ((VecHandle*)p)->env->probe(env_index, (int32_t*)out);
}
int bcv_last_deaths(void* p, int env_index, int* out, int cap) {
    return ((VecHandle*)p)->env->last_deaths(env_index, (int32_t*)out, cap);
}
int bcv_last_splits(void* p, int env_index, int* out, int cap) {
    return ((VecHandle*)p)->env->last_splits(env_index, (int32_t*)out, cap);
}
int bcv_probe_fields() { return VecEnv::PROBE_FIELDS; }

// Perfect-play labels (VecEnv::perfect, supervised_learning.md) for every env's acting dragon:
// the kind into kind_out[env], the correct actions as CODEC_ACTIONS bytes per env into acts_out.
// Returns how many envs have one. Read-only.
// A synthetic position in env i (VecEnv::set_scenario): 1 on success, 0 if rejected.
int bcv_set_scenario(void* p, int env_index, int round, int n, const int* team, const int* len,
                     const int* cells, int n_pearls, const int* pearls, int acting) {
    return ((VecHandle*)p)->env->set_scenario(env_index, round, n, team, len, cells, n_pearls, pearls,
                                              acting) ? 1 : 0;
}

// The far sprints' targets (BC_FARSPRINT): CODEC_FAR (ox, oy) pairs, the first id; returns the count.
int bcv_far_targets(int* out, int* first_id) {
    for (int k = 0; k < CODEC_FAR; k++) { out[2 * k] = FAR_TARGETS.ox[k]; out[2 * k + 1] = FAR_TARGETS.oy[k]; }
    *first_id = CODEC_FAR_ID;
    return CODEC_FAR;
}

// Test hooks: far sprint k's path for env i's acting dragon (dirs as 0 N 1 E 2 S 3 W; returns
// its length, 0 = none), and a dragon's head cell (-1 when dead).
int bcv_far_path(void* p, int env_index, int k, int* dirs_out) {
    VecEnv* v = ((VecHandle*)p)->env;
    char dirs[MAX_STEPS];
    const int n = v->far_path_of(env_index, k, dirs);
    for (int s = 0; s < n; s++) dirs_out[s] = dir_index(dirs[s]);
    return n;
}
int bcv_dragon_head(void* p, int env_index, int dragon_id) { return ((VecHandle*)p)->env->dragon_head(env_index, dragon_id); }
void bcv_env_maps(void* p, int* out) {
    auto* h = (VecHandle*)p;
    for (int i = 0; i < h->env->num_envs(); i++) out[i] = h->env->env_map(i);
}
int bcv_dragon_len(void* p, int env_index, int dragon_id) { return ((VecHandle*)p)->env->dragon_len(env_index, dragon_id); }

int bcv_perfect(void* p, int* kind_out, unsigned char* acts_out) {
    VecEnv* v = ((VecHandle*)p)->env;
    int n = 0;
    for (int i = 0; i < v->num_envs(); i++) {
        kind_out[i] = v->perfect(i, acts_out + (size_t)i * CODEC_ACTIONS);
        n += kind_out[i] != VecEnv::PP_NONE;
    }
    return n;
}

int bcv_acting_dragon(void* p, int env_index) {
    return ((VecHandle*)p)->env->acting_dragon_id(env_index);
}

int bcv_round_block(void* p, int env_index, char* buf, int cap) {
    const std::string s = ((VecHandle*)p)->env->round_block(env_index);
    const int n = (int)s.size();
    if (buf && cap > 0) memcpy(buf, s.data(), n < cap ? n : cap);
    return n;
}

void bcv_set_pearl_seed64(void* p, int env, unsigned long long seed) {
    ((VecHandle*)p)->env->set_pearl_seed64(env, (uint64_t)seed);
}

void bcv_set_map_weights(void* p, const double* w) {
    ((VecHandle*)p)->env->set_map_weights(w);
}

void bcv_set_env_opponent(void* p, int env_index, int team, int bot_kind, int fixed_map) {
    ((VecHandle*)p)->env->set_env_opponent(env_index, team, bot_kind, fixed_map);
}

void bcv_set_potential_gamma(void* p, float g) { ((VecHandle*)p)->env->set_potential_gamma(g); }
void bcv_set_queen_guard(void* p, int on) { ((VecHandle*)p)->env->set_queen_guard(on != 0); }
void bcv_set_queen_deadend(void* p, int level) { ((VecHandle*)p)->env->set_queen_deadend(level); }
void bcv_set_reward_v8(void* p, int on, float kappa, int credit) {
    ((VecHandle*)p)->env->set_reward_v8(on != 0, kappa, credit);
}

int bcv_ep_cols() { return EP_COLS; }
int bcv_bot_count() { return BOT_COUNT; }
const char* bcv_bot_name(int k) { return bot_name(k); }

// Sizes, so python never hard codes a layout.
void bcv_layout(int* out) {
    out[0] = LC_COUNT;
    out[1] = WINDOW;
    out[2] = SC_TOTAL;   // 14 base + mem + memfar
    out[3] = MAX_MSGS;
    out[4] = CODEC_ACTIONS;
    out[5] = RW_COUNT;
    out[6] = MAX_STEPS;
    out[7] = CODEC_MOVES;
    out[8] = SONAR_DIRS;
    out[9] = NUM_MSGS_CAP;
}

}  // extern "C"
