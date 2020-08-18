#ifndef _VESINTERACTION_H_
#define _VESINTERACTION_H_

#include <typeinfo>
#include "Enums.h"
#include "Error.h"
#include <omp.h>
/**
 * The gateway function between the local code and FMM code. This
 * class takes care of multi-threads, organizing date, copying data to
 * the host, etc. It is a template to avoid the requirement that the
 * <tt>InteractionFun_t</tt> match the data type of
 * <tt>VecContainer</tt> (since template typedefs are not legal in
 * C++).
 */
template<typename T>
class VesInteraction
{
  public:
    ///The pointer-type to the external FMM function handle.
    typedef void(*InteractionFun_t)(const T*, const T*, size_t, T*, void**);

    //Deallocator for the context
    typedef void(*Dealloc_t)(void**);

    /**
     * @param interaction_handle The function pointer to the FMM
     * code. When set to <tt>NULL</tt>, no interaction is performed.
     * @param num_threads The expected number of threads this class
     * need to take care of.
     */
    explicit VesInteraction(InteractionFun_t interaction_handle = NULL,
        Dealloc_t clear_context = NULL,
        int num_threads = omp_get_max_threads());
    ~VesInteraction();

    /**
     * The function called inside the local code to perform
     * interaction. Each thread called this method to perform
     * interaction with other threads and/or other MPI processes.
     *
     * @param position The Cartesian coordinate of the points.
     * @param density The density at each point.
     * @param potential (return value) The calculated potential on the points.
     *
     * @return Enum type defined in enums.h indicating the outcome of interaction.
     */
    template<typename VecContainer>
    Error_t operator()(const VecContainer &position, VecContainer &density,
        VecContainer &potential) const;

    bool HasInteraction() const;
  private:
    InteractionFun_t interaction_handle_;
    Dealloc_t clear_context_;

    int num_threads_;
    size_t* each_thread_np_;
    size_t* each_thread_idx_;

    mutable size_t np_;
    mutable size_t containers_capacity_;

    mutable T* all_pos_;
    mutable T* all_den_;
    mutable T* all_pot_;
    mutable void* context_;

    size_t getCpyDestIdx(size_t this_thread_np) const;
    void checkContainersSize() const;
};

#include "VesInteraction.cc"

#endif // _VESINTERACTION_H_
