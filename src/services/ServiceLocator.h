#pragma once

#include <memory>

#include "services/IDataAccessService.h"

namespace sep::infra {

/**
 * Simple service locator providing access to IDataAccessService
 */
class ServiceLocator {
public:
    static void provide(std::shared_ptr<IDataAccessService> service) {
        instance() = std::move(service);
    }

    static std::shared_ptr<IDataAccessService> dataAccess() {
        return instance();
    }

private:
    static std::shared_ptr<IDataAccessService>& instance() {
        static std::shared_ptr<IDataAccessService> svc;
        return svc;
    }
};

} // namespace sep::infra

