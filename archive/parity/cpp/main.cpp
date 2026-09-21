// Dumps a checksum of the observation and the mask each turn, and plays the
// first legal action so the trajectory is deterministic. The Python bot in
// ../py does the same; identical logs mean the two ports agree.
#include "helper.hpp"
#include "obs.hpp"
#include <cstdint>
#include <cstdio>

int main() {
    auto [ct, game] = unswbc::init();
    obs::Snapshot snap;
    float local[obs::N_CHANNELS * obs::CELLS];
    float sc[obs::N_SCALARS];
    std::uint8_t mask[obs::N_ACTIONS];

    while (unswbc::update(ct, game)) {
        snap.build(ct, game);
        snap.mask(ct, game, mask);
        snap.local(local);
        snap.scalars(ct, game, sc);

        long long lsum = 0;
        for (int i = 0; i < obs::N_CHANNELS * obs::CELLS; i++)
            lsum += (long long)(local[i] * 1000.0f + 0.5f) * (i + 1);
        long long ssum = 0;
        for (int i = 0; i < obs::N_SCALARS; i++)
            ssum += (long long)(sc[i] * 1000.0f + 0.5f) * (i + 1);
        char bits[obs::N_ACTIONS + 1];
        for (int i = 0; i < obs::N_ACTIONS; i++) bits[i] = mask[i] ? '1' : '0';
        bits[obs::N_ACTIONS] = '\0';

        int act = -1;
        for (int i = 0; i < obs::N_ACTIONS; i++) if (mask[i]) { act = i; break; }
        if (act < 0) act = 0;

        ct.output_log("P", ct.get_id(), game.round_num, bits, lsum, ssum, act);
        if (game.round_num <= 1) {
            for (int ch = 0; ch < obs::N_CHANNELS; ch++) {
                long long s = 0;
                for (int j = 0; j < obs::CELLS; j++)
                    s += (long long)(local[ch * obs::CELLS + j] * 1000.0f + 0.5f) * (j + 1);
                std::printf("LOG C %d %d %d %lld\n", ct.get_id(), game.round_num, ch, s);
            }
        }

        if (act < obs::N_MOVES) {
            std::array<int, 3> dirs{};
            int const n = obs::decode_move(act, snap.facing, dirs);
            std::vector<unswbc::Direction> out;
            for (int i = 0; i < n; i++) out.push_back(obs::dir_of(dirs[(std::size_t)i]));
            ct.make_moves(out);
        } else {
            int const k = obs::SPLIT_K[(std::size_t)(act - obs::N_MOVES)];
            ct.do_split(k < 0 ? snap.length / 2 : k);
        }
        unswbc::end_turn();
    }
}
