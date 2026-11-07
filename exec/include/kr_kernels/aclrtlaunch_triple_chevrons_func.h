
#ifndef HEADER_ACLRTLAUNCH_KR_PUBLISH_AIV_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_KR_PUBLISH_AIV_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_kr_publish_aiv(uint32_t numBlocks, void* stream, void* ctrGM, uint32_t index);

inline uint32_t kr_publish_aiv(uint32_t numBlocks, void* hold, void* stream, void* ctrGM, uint32_t index)
{
    (void)hold;
    return aclrtlaunch_kr_publish_aiv(numBlocks, stream, ctrGM, index);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_KR_WAIT_AIV_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_KR_WAIT_AIV_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_kr_wait_aiv(uint32_t numBlocks, void* stream, void* ctrGM, uint32_t index, uint32_t maxSpins);

inline uint32_t kr_wait_aiv(uint32_t numBlocks, void* hold, void* stream, void* ctrGM, uint32_t index, uint32_t maxSpins)
{
    (void)hold;
    return aclrtlaunch_kr_wait_aiv(numBlocks, stream, ctrGM, index, maxSpins);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_KR_PUBLISH_AIC_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_KR_PUBLISH_AIC_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_kr_publish_aic(uint32_t numBlocks, void* stream, void* ctrGM, uint32_t index);

inline uint32_t kr_publish_aic(uint32_t numBlocks, void* hold, void* stream, void* ctrGM, uint32_t index)
{
    (void)hold;
    return aclrtlaunch_kr_publish_aic(numBlocks, stream, ctrGM, index);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_KR_WAIT_AIC_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_KR_WAIT_AIC_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_kr_wait_aic(uint32_t numBlocks, void* stream, void* ctrGM, uint32_t index, uint32_t maxSpins);

inline uint32_t kr_wait_aic(uint32_t numBlocks, void* hold, void* stream, void* ctrGM, uint32_t index, uint32_t maxSpins)
{
    (void)hold;
    return aclrtlaunch_kr_wait_aic(numBlocks, stream, ctrGM, index, maxSpins);
}

#endif
