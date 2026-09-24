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
// Broadcast sonar in all four directions every turn and speak protocol 3.
// A setter rather than another bcv_create argument: the signature is shared
// with the replay and privileged builds and with every existing caller.
void bcv_set_sonar(void* p, int on) { ((VecHandle*)p)->env->set_sonar(on != 0); }

void bcv_bind_board(void* p, uint8_t* board) { ((VecHandle*)p)->env->bind_board(board); }
void bcv_board_shape(int* out) { out[0] = BOARD_CH; out[1] = BOARD_MAX; }

void bcv_bind_wide(void* p, float* wide) { ((VecHandle*)p)->env->bind_wide(wide); }
void bcv_wide_shape(int* out) { out[0] = 2 * wide_cfg::CH; out[1] = wide_cfg::SIDE; }

void bcv_destroy(void* p) {
    auto* h = (VecHandle*)p;
    delete h->env;
    delete h;
}

void bcv_bind(void* p, float* local, float* scalar, unsigned int* msgs, unsigned char* mask,
              long long* uid, int* dragon_id, signed char* team, int* round_out) {
    ((VecHandle*)p)->env->bind(local, scalar, msgs, mask, (int64_t*)uid, (int32_t*)dragon_id,
                               (int8_t*)team, (int32_t*)round_out);
}

void bcv_reset(void* p) { ((VecHandle*)p)->env->reset(); }

// Structured actions, one per env, as flat arrays.
void bcv_step(void* p, const signed char* kind, const signed char* n_steps,
              const signed char* dirs, const short* split_k,
              const signed char* send_sonar, const unsigned int* sonar) {
    auto* h = (VecHandle*)p;
    const int n = h->env->num_envs();
    for (int i = 0; i < n; i++) {
        Action& a = h->actions[i];
        a.kind = kind[i];
        a.n_steps = n_steps[i];
        memcpy(a.dirs, dirs + (size_t)i * MAX_STEPS, MAX_STEPS);
        a.split_k = split_k[i];
        a.send_sonar = send_sonar[i];
        a.sonar = sonar[i];
    }
    h->env->step(h->actions.data());
}

// Codec actions: one id per env, plus the optional sonar payload.
void bcv_step_codec(void* p, const int* action_ids, const signed char* send_sonar,
                    const unsigned int* sonar) {
    auto* h = (VecHandle*)p;
    const int n = h->env->num_envs();
    for (int i = 0; i < n; i++) {
        Action a = h->env->decode(i, action_ids[i]);
        a.send_sonar = send_sonar ? send_sonar[i] : 0;
        a.sonar = sonar ? sonar[i] : 0u;
        h->actions[i] = a;
    }
    h->env->step(h->actions.data());
}

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
int bcv_probe_fields() { return VecEnv::PROBE_FIELDS; }

int bcv_acting_dragon(void* p, int env_index) {
    return ((VecHandle*)p)->env->acting_dragon_id(env_index);
}

int bcv_round_block(void* p, int env_index, char* buf, int cap) {
    const std::string s = ((VecHandle*)p)->env->round_block(env_index);
    const int n = (int)s.size();
    if (buf && cap > 0) memcpy(buf, s.data(), n < cap ? n : cap);
    return n;
}

void bcv_set_map_weights(void* p, const double* w) {
    ((VecHandle*)p)->env->set_map_weights(w);
}

void bcv_set_env_opponent(void* p, int env_index, int team, int bot_kind, int fixed_map) {
    ((VecHandle*)p)->env->set_env_opponent(env_index, team, bot_kind, fixed_map);
}

void bcv_set_potential_gamma(void* p, float g) { ((VecHandle*)p)->env->set_potential_gamma(g); }
void bcv_set_reward_v8(void* p, int on, float kappa) {
    ((VecHandle*)p)->env->set_reward_v8(on != 0, kappa);
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
}

}  // extern "C"
