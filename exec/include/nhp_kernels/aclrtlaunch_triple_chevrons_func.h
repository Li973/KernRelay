
#ifndef HEADER_ACLRTLAUNCH_PUBLISH_TILE_FLAG_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_PUBLISH_TILE_FLAG_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_publish_tile_flag(uint32_t numBlocks, void* stream, void* flagsGM, uint32_t tile);

inline uint32_t publish_tile_flag(uint32_t numBlocks, void* hold, void* stream, void* flagsGM, uint32_t tile)
{
    (void)hold;
    return aclrtlaunch_publish_tile_flag(numBlocks, stream, flagsGM, tile);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_NHP_AIV_DELAY_PUBLISH_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_NHP_AIV_DELAY_PUBLISH_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_nhp_aiv_delay_publish(uint32_t numBlocks, void* stream, void* flagGM, uint32_t index, uint32_t delayIters);

inline uint32_t nhp_aiv_delay_publish(uint32_t numBlocks, void* hold, void* stream, void* flagGM, uint32_t index, uint32_t delayIters)
{
    (void)hold;
    return aclrtlaunch_nhp_aiv_delay_publish(numBlocks, stream, flagGM, index, delayIters);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_NHP_AIV_SCALE_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_NHP_AIV_SCALE_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_nhp_aiv_scale(uint32_t numBlocks, void* stream, void* inGM, void* outGM, uint32_t elems, uint32_t iters);

inline uint32_t nhp_aiv_scale(uint32_t numBlocks, void* hold, void* stream, void* inGM, void* outGM, uint32_t elems, uint32_t iters)
{
    (void)hold;
    return aclrtlaunch_nhp_aiv_scale(numBlocks, stream, inGM, outGM, elems, iters);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_NHP_AIV_SPIN_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_NHP_AIV_SPIN_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_nhp_aiv_spin(uint32_t numBlocks, void* stream, void* flagGM, uint32_t index, void* statusGM, uint32_t noWait);

inline uint32_t nhp_aiv_spin(uint32_t numBlocks, void* hold, void* stream, void* flagGM, uint32_t index, void* statusGM, uint32_t noWait)
{
    (void)hold;
    return aclrtlaunch_nhp_aiv_spin(numBlocks, stream, flagGM, index, statusGM, noWait);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_NHP_AIC_MAC_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_NHP_AIC_MAC_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_nhp_aic_mac(uint32_t numBlocks, void* stream, void* inGM, void* outGM, uint32_t elems, uint32_t iters);

inline uint32_t nhp_aic_mac(uint32_t numBlocks, void* hold, void* stream, void* inGM, void* outGM, uint32_t elems, uint32_t iters)
{
    (void)hold;
    return aclrtlaunch_nhp_aic_mac(numBlocks, stream, inGM, outGM, elems, iters);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_NHP_AIC_DELAY_PUBLISH_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_NHP_AIC_DELAY_PUBLISH_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_nhp_aic_delay_publish(uint32_t numBlocks, void* stream, void* flagGM, uint32_t index, uint32_t delayIters);

inline uint32_t nhp_aic_delay_publish(uint32_t numBlocks, void* hold, void* stream, void* flagGM, uint32_t index, uint32_t delayIters)
{
    (void)hold;
    return aclrtlaunch_nhp_aic_delay_publish(numBlocks, stream, flagGM, index, delayIters);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_NHP_AIC_SPIN_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_NHP_AIC_SPIN_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_nhp_aic_spin(uint32_t numBlocks, void* stream, void* flagGM, uint32_t index, void* statusGM, uint32_t noWait);

inline uint32_t nhp_aic_spin(uint32_t numBlocks, void* hold, void* stream, void* flagGM, uint32_t index, void* statusGM, uint32_t noWait)
{
    (void)hold;
    return aclrtlaunch_nhp_aic_spin(numBlocks, stream, flagGM, index, statusGM, noWait);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_NHP_MIX_SCALE_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_NHP_MIX_SCALE_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_nhp_mix_scale(uint32_t numBlocks, void* stream, void* inGM, void* outGM, uint32_t elems, uint32_t iters);

inline uint32_t nhp_mix_scale(uint32_t numBlocks, void* hold, void* stream, void* inGM, void* outGM, uint32_t elems, uint32_t iters)
{
    (void)hold;
    return aclrtlaunch_nhp_mix_scale(numBlocks, stream, inGM, outGM, elems, iters);
}

#endif
